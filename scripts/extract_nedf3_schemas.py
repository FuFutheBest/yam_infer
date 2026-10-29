"""Copy the protobuf schemas embedded in an SI NEDF3 recording into yam_infer/data/nedf3_schemas/.

    .venv/bin/python scripts/extract_nedf3_schemas.py <SI episode dir>

Every SuperInference NEDF3 .mcap carries the FileDescriptorSet of each message type it uses
(nmx.msg.Metadata / ArmState / HandState / RGBD).  We write our recordings with exactly these
schemas, so no private nmx_msg / dm3_core package is needed.  Only needed again if SI changes them.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcap.reader import make_reader

NAMES = ("nmx.msg.Metadata", "nmx.msg.ArmState", "nmx.msg.HandState", "nmx.msg.RGBD")
OUT = Path(__file__).resolve().parents[1] / "yam_infer" / "data" / "nedf3_schemas"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    files = sorted((Path(sys.argv[1]) / "data").glob("*.mcap"))
    if not files:
        raise SystemExit(f"no data/*.mcap under {sys.argv[1]}")
    try:
        with open(files[0], "rb") as fh:
            summary = make_reader(fh).get_summary()
    except OSError as exc:
        raise SystemExit(f"cannot read {files[0]}: {exc}") from exc
    if summary is None:
        raise SystemExit(f"{files[0]} has no summary section")
    found = {s.name: s for s in summary.schemas.values() if s.encoding == "protobuf"}
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        if name not in found:
            raise SystemExit(f"schema {name} not in {files[0]}")
        (OUT / f"{name}.desc").write_bytes(found[name].data)
        print(f"{name}: {len(found[name].data)} bytes -> {OUT / (name + '.desc')}")


if __name__ == "__main__":
    main()
