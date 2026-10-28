"""Optional remote inference, for when the robot PC has no usable GPU.

GPU machine :  python -m yampilot.server --ckpt model.pt --port 5555
Robot PC    :  python -m yampilot.run ... --remote GPU_HOST:5555

Protocol (no pickle): 4-byte header length + JSON header + raw binary blobs listed in the header.
The robot side does the exact training preprocessing (JPEG round-trip + resize) and
sends the small 240x320 images as JPEG, so bandwidth is ~30 KB per request.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct

import cv2
import numpy as np


def _send(sock: socket.socket, header: dict, blobs: list[bytes] | None = None) -> None:
    blobs = blobs or []
    header = {**header, "_blob_sizes": [len(b) for b in blobs]}
    head = json.dumps(header).encode()
    sock.sendall(struct.pack("!I", len(head)) + head + b"".join(blobs))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise ConnectionError("socket closed")
        buf += part
    return bytes(buf)


def _recv(sock: socket.socket) -> tuple[dict, list[bytes]]:
    (n,) = struct.unpack("!I", _recv_exact(sock, 4))
    if n > 16 * 1024 * 1024:
        raise ConnectionError("header too large")
    header = json.loads(_recv_exact(sock, n))
    blobs = [_recv_exact(sock, int(size)) for size in header.pop("_blob_sizes", [])]
    return header, blobs


def _encode(images: np.ndarray) -> list[bytes]:
    out = []
    for img in images:
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        out.append(buf.tobytes())
    return out


def _decode(blobs: list[bytes]) -> np.ndarray:
    imgs = []
    for b in blobs:
        bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("JPEG decode failed")
        imgs.append(np.asarray(bgr)[:, :, ::-1])
    return np.ascontiguousarray(np.stack(imgs))


class RemotePolicy:
    """Same interface as yampilot.policy.Policy, but the model runs on a server."""

    def __init__(self, address: str, jpeg_roundtrip: bool = True, timeout: float = 2.0):
        host, port = address.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _send(self.sock, {"cmd": "info"})
        self._info, _ = _recv(self.sock)
        self._info["device"] = f"remote {address} ({self._info['device']})"
        self.jpeg_roundtrip = jpeg_roundtrip
        self.cameras = self._info["cameras"]
        self.image_size = tuple(self._info["image_size"])
        self.fps = float(self._info["fps"])
        self.chunk_size = int(self._info["chunk_size"])
        self.a_min = np.asarray(self._info["action_min"])
        self.a_max = np.asarray(self._info["action_max"])

    def info(self) -> dict:
        return self._info

    def predict(self, frames: dict[str, np.ndarray], qpos: np.ndarray, task: str):
        from yampilot.policy import preprocess_frame

        images = np.stack([preprocess_frame(frames[c], self.image_size, self.jpeg_roundtrip) for c in self.cameras])
        _send(self.sock, {"cmd": "predict", "qpos": np.asarray(qpos, float).tolist(), "task": task},
              _encode(images))
        reply, _ = _recv(self.sock)
        if "error" in reply:
            raise RuntimeError(f"server error: {reply['error']}")
        return np.asarray(reply["actions"], dtype=np.float64), images


def main() -> None:
    from yampilot.policy import Policy

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="127.0.0.1", help="use the LAN IP to accept the robot PC")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    policy = Policy(args.ckpt, args.device)
    srv = socket.create_server((args.host, args.port))
    print(f"policy server on {args.host}:{args.port} ({policy.device})", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"client {addr}", flush=True)
        with conn:
            try:
                while True:
                    msg, blobs = _recv(conn)
                    if msg.get("cmd") == "info":
                        _send(conn, policy.info())
                    elif msg.get("cmd") == "predict":
                        try:
                            qpos = np.asarray(msg["qpos"], dtype=np.float64)
                            actions = policy.predict_prepared(_decode(blobs), qpos, str(msg["task"]))
                            _send(conn, {"actions": actions.tolist()})
                        except Exception as exc:  # noqa: BLE001 - report to client
                            _send(conn, {"error": repr(exc)})
            except (ConnectionError, json.JSONDecodeError, KeyError):
                print(f"client {addr} disconnected", flush=True)


if __name__ == "__main__":
    main()
