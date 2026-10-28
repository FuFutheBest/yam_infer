"""Find (and interactively stop) whatever keeps this project from opening the robot.

    python -m yam_infer.tools.free_hw            # list, ask, stop, re-check   (exit 0 = free)
    python -m yam_infer.tools.free_hw --check    # list only                    (exit 1 = busy)
    python -m yam_infer.tools.free_hw --sudo     # also inspect other users' processes

Looks at the resources themselves, not at program names:

| resource        | how                                                                  |
|-----------------|----------------------------------------------------------------------|
| CAN buses       | open sockets whose protocol is CAN_* (getxattr system.sockprotoname) |
| RPC ports       | LISTEN sockets on the follower/leader ports of configs/hardware.yaml |
| cameras         | open /dev/video* or /dev/bus/usb/* belonging to an Intel (RealSense) |
| CAN interfaces  | follower/leader interfaces of the config that are not UP             |

Holders are grouped by their "owner": the top-most ancestor that is a real program (not a
shell / terminal / systemd), e.g. the 4 SI workers + 2 camera processes -> the SI WebUI.
Stopping = SIGINT (like Ctrl+C), then SIGTERM, then SIGKILL.  Other users' processes are
stopped with sudo (sudo asks for the password itself).
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HW_CONFIG = PROJECT_ROOT / "configs" / "hardware.yaml"
SHELLS = {"bash", "sh", "dash", "zsh", "fish", "tcsh", "csh", "ksh"}
INFRA = {
    "systemd",
    "init",
    "sshd",
    "sudo",
    "su",
    "login",
    "tmux",
    "tmux: server",
    "screen",
    "gnome-terminal-server",
    "gnome-terminal-",
    "konsole",
    "xterm",
    "kitty",
    "alacritty",
    "terminator",
    "tilix",
    "code",
    "nohup",
    "setsid",
    "gnome-shell",
    "gdm-session-wor",
}
INTEL_VENDOR = "8086"
OUR_MODULES = ("yam_infer.run", "yam_infer.hw.follower.worker")


# ----------------------------------------------------------------------------- /proc helpers
def _read(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def _cmdline(pid: int) -> str:
    return _read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()


def python_module(pid: int) -> str:
    """`X` if the process is a real `python... -m X ...` (never a shell that mentions it)."""
    argv = _read(f"/proc/{pid}/cmdline").split("\0")
    if not argv or not re.search(r"python[0-9.]*$", argv[0]):
        return ""
    i = 1
    while i < len(argv) - 1 and argv[i].startswith("-"):  # interpreter options: -u, -X opt, ...
        if argv[i] == "-m":
            return argv[i + 1]
        i += 2 if argv[i] in ("-X", "-W", "-Q") else 1
    return ""


def is_ours(pid: int, module: str | None = None) -> bool:
    mod = python_module(pid)
    return mod in OUR_MODULES and (module is None or mod == module)


def _comm(pid: int) -> str:
    return _read(f"/proc/{pid}/comm").strip()


def _ppid(pid: int) -> int:
    stat = _read(f"/proc/{pid}/stat")
    try:
        return int(stat[stat.rindex(")") + 2 :].split()[1])
    except (ValueError, IndexError):
        return 0


def _uid(pid: int) -> int:
    for line in _read(f"/proc/{pid}/status").splitlines():
        if line.startswith("Uid:"):
            return int(line.split()[1])
    return -1


def _user(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _ancestors(pid: int) -> list[int]:
    chain, seen = [], set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        chain.append(pid)
        pid = _ppid(pid)
    return chain


def _is_wrapper(pid: int) -> bool:
    """Shells, terminals, ssh, systemd... - never the 'program' that owns a device."""
    comm = _comm(pid)
    argv0 = os.path.basename((_cmdline(pid).split() or [""])[0]).lstrip("-")
    return comm in INFRA or comm in SHELLS or argv0 in SHELLS or argv0 in INFRA or pid <= 1


def owner_of(pid: int) -> int:
    """Top-most ancestor that is still a real program (stops at the first shell/terminal)."""
    owner = pid
    for p in _ancestors(pid)[1:]:
        if _is_wrapper(p):
            break
        owner = p
    return owner


def _usb_vendor_of_video(dev: str) -> str:
    node = Path("/sys/class/video4linux") / os.path.basename(dev) / "device"
    for parent in [node.resolve(), *node.resolve().parents][:4]:
        v = _read(str(parent / "idVendor")).strip()
        if v:
            return v
    return ""


def _usb_vendor_of_bus(dev: str) -> str:
    m = re.match(r"/dev/bus/usb/(\d+)/(\d+)$", dev)
    if not m:
        return ""
    bus, num = int(m.group(1)), int(m.group(2))
    for d in Path("/sys/bus/usb/devices").glob("*"):
        if _read(str(d / "busnum")).strip() == str(bus) and _read(str(d / "devnum")).strip() == str(
            num
        ):
            return _read(str(d / "idVendor")).strip()
    return ""


def _listen_inodes(ports: set[int]) -> dict[int, int]:
    """socket inode -> port, for LISTEN sockets on the given ports (tcp + tcp6)."""
    found = {}
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in _read(table).splitlines()[1:]:
            f = line.split()
            if len(f) < 10 or f[3] != "0A":  # 0A = LISTEN
                continue
            port = int(f[1].rsplit(":", 1)[1], 16)
            if port in ports:
                found[int(f[9])] = port
    return found


def _client_inodes(ports: set[int]) -> dict[int, int]:
    """socket inode -> port, for ESTABLISHED connections TO the given ports (RPC clients)."""
    found = {}
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in _read(table).splitlines()[1:]:
            f = line.split()
            if len(f) < 10 or f[3] != "01":  # 01 = ESTABLISHED
                continue
            port = int(f[2].rsplit(":", 1)[1], 16)  # remote side = the worker's port
            if port in ports and int(f[9]):
                found[int(f[9])] = port
    return found


def _can_receivers() -> set[str]:
    """CAN interfaces with at least one kernel receiver (someone has a bound CAN socket)."""
    devs = set()
    for line in _read("/proc/net/can/rcvlist_all").splitlines():
        f = line.split()
        if f and f[0].startswith(("can", "vcan", "slcan")) and len(f) >= 4:
            devs.add(f[0])
    return devs


# ----------------------------------------------------------------------------- scan
@dataclass
class Scan:
    holders: dict[int, list[str]] = field(default_factory=dict)  # pid -> reasons
    unreadable: int = 0  # processes whose fds we could not inspect (other users)
    ports_unexplained: list[int] = field(default_factory=list)
    can_unexplained: list[str] = field(default_factory=list)


def load_hw(hw_config: Path) -> tuple[dict[int, str], list[str]]:
    cfg = yaml.safe_load(Path(hw_config).read_text())
    ports, ifaces = {}, []
    for side, arm in cfg["arms"].items():
        for role in ("follower", "leader"):
            ports[int(arm[role]["rpc_port"])] = f"{side} {role}"
            ifaces.append(arm[role]["can_channel"])
    return ports, ifaces


def scan(hw_config: Path, ports_only: bool = False) -> Scan:
    """ports_only: only the RPC ports matter (fake/simulated workers, tests)."""
    ports, _ = load_hw(hw_config)
    listen = _listen_inodes(set(ports))
    clients = _client_inodes(set(ports))
    res = Scan()
    me = set(_ancestors(os.getpid()))
    seen_ports = set()
    any_can_holder = False
    try:
        entries = os.listdir("/proc")
    except OSError as exc:
        raise RuntimeError(f"cannot list /proc: {exc}") from exc
    for name in entries:
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in me:
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except PermissionError:
            res.unreadable += 1
            continue
        except OSError:
            continue
        reasons: set[str] = set()
        can = 0
        for fd in fds:
            path = f"/proc/{pid}/fd/{fd}"
            try:
                target = os.readlink(path)
            except OSError:
                continue
            if target.startswith("socket:["):
                inode = int(target[8:-1])
                if inode in clients:
                    reasons.add(
                        f"连着 RPC 端口 {clients[inode]}（在控制/监视 {ports[clients[inode]]}）"
                    )
                if inode in listen:
                    reasons.add(f"RPC 端口 {listen[inode]}（{ports[listen[inode]]}）")
                    seen_ports.add(listen[inode])
                try:
                    proto = os.getxattr(path, "system.sockprotoname").decode().strip("\0")
                except OSError:
                    proto = ""
                if proto.startswith("CAN"):
                    can += 1
            elif target.startswith("/dev/video"):
                if _usb_vendor_of_video(target) == INTEL_VENDOR:
                    reasons.add(f"相机 {target}")
            elif target.startswith("/dev/bus/usb/"):
                if _usb_vendor_of_bus(target) == INTEL_VENDOR:
                    reasons.add(f"相机 USB {target}")
        if ports_only:
            reasons = {r for r in reasons if "RPC" in r}
            can = 0
        if can:
            any_can_holder = True
            reasons.add(f"CAN 总线（{can} 个 CAN socket）")
        if reasons:
            res.holders[pid] = sorted(reasons)
    res.ports_unexplained = sorted(set(listen.values()) - seen_ports)
    if not any_can_holder and not ports_only:
        _, ifaces = load_hw(hw_config)
        res.can_unexplained = sorted(_can_receivers() & set(ifaces))
    return res


def down_ifaces(hw_config: Path) -> list[str]:
    _, ifaces = load_hw(hw_config)
    down = []
    for i in ifaces:
        flags = _read(f"/sys/class/net/{i}/operstate").strip()
        if flags != "up":
            down.append(f"{i}({flags or '不存在'})")
    return down


def scan_as_root(hw_config: Path) -> Scan | None:
    print("用 sudo 检查其他用户的进程（sudo 会询问密码）...")
    cmd = [
        "sudo",
        "-p",
        "[sudo] 请输入 %u 的密码: ",
        sys.executable,
        "-m",
        "yam_infer.tools.free_hw",
        "--scan-json",
        "--hw-config",
        str(hw_config),
    ]
    try:
        out = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, text=True, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"  sudo 检查失败: {exc}")
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError as exc:
        print(f"  sudo 检查输出无法解析: {exc}")
        return None
    return Scan(
        {int(k): v for k, v in data["holders"].items()},
        0,
        data["ports_unexplained"],
        data["can_unexplained"],
    )


# ----------------------------------------------------------------------------- report / stop
@dataclass
class Group:
    owner: int
    members: dict[int, list[str]]

    @property
    def uid(self) -> int:
        return _uid(self.owner)

    @property
    def ours(self) -> bool:
        return is_ours(self.owner)


def group(holders: dict[int, list[str]]) -> list[Group]:
    groups: dict[int, Group] = {}
    for pid, reasons in holders.items():
        o = owner_of(pid)
        groups.setdefault(o, Group(o, {})).members[pid] = reasons
    return sorted(groups.values(), key=lambda g: g.owner)


def describe(groups: list[Group]) -> None:
    for i, g in enumerate(groups, 1):
        cmd = _cmdline(g.owner) or _comm(g.owner)
        tag = "  [本项目]" if g.ours else ""
        print(f"\n  [{i}] pid {g.owner}  用户 {_user(g.uid)}{tag}\n      {cmd[:150]}")
        for pid, reasons in sorted(g.members.items()):
            who = "本进程" if pid == g.owner else f"子进程 {pid}"
            print(f"      - {who}: {'; '.join(reasons)}")
        if any(r.startswith("CAN") for rs in g.members.values() for r in rs):
            print(
                "      ! 它在控制机械臂：停止后电机断力，手臂会落下（想先归位：bash scripts/home_arms.sh）"
            )


def _alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists() and "zombie" not in _read(f"/proc/{pid}/status").lower()


def _signal(pid: int, sig: signal.Signals, use_sudo: bool) -> None:
    if use_sudo:
        subprocess.run(
            ["sudo", "-p", "[sudo] 请输入 %u 的密码: ", "kill", f"-{sig.name[3:]}", str(pid)],
            check=False,
        )
    else:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def stop(g: Group) -> bool:
    """SIGINT -> SIGTERM -> SIGKILL on the owner (its children are expected to follow)."""
    use_sudo = g.uid != os.getuid() and os.getuid() != 0
    pids = [g.owner, *[p for p in g.members if p != g.owner]]
    waits = [
        (signal.SIGINT, 45.0 if g.ours else 10.0),
        (signal.SIGTERM, 8.0),
        (signal.SIGKILL, 3.0),
    ]
    for sig, wait in waits:
        targets = [p for p in pids if _alive(p)] if sig != signal.SIGINT else [g.owner]
        if not targets:
            break
        for p in targets:
            print(f"  -> {sig.name} pid {p}")
            _signal(p, sig, use_sudo)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.2)
        if not any(_alive(p) for p in pids):
            break
    left = [p for p in pids if _alive(p)]
    if left:
        print(f"  !! 仍在运行: {left}")
    return not left


def ask(prompt: str, default: str = "n") -> str:
    try:
        return (input(prompt).strip() or default).lower()
    except EOFError:
        return default


def choose(groups: list[Group]) -> list[Group]:
    ans = ask("\n停止哪些？ a=全部  1,2=编号  n=都不停 [n]: ")
    if ans in ("a", "all", "y", "yes"):
        return groups
    picked = []
    for tok in re.split(r"[,\s]+", ans):
        if tok.isdigit() and 1 <= int(tok) <= len(groups):
            picked.append(groups[int(tok) - 1])
    return picked


def free(
    hw_config: Path,
    interactive: bool = True,
    use_sudo: bool = False,
    ports_only: bool = False,
    ignore_pids: set[int] | None = None,
) -> bool:
    """True when nothing blocks the robot any more.

    ports_only: only RPC ports matter (fake workers); ignore_pids: holders the caller keeps using.
    """
    asked_sudo = False
    for _round in range(5):
        res = scan_as_root(hw_config) if use_sudo else scan(hw_config, ports_only)
        if res is None:
            res = scan(hw_config, ports_only)
        unexplained = res.ports_unexplained or res.can_unexplained
        if unexplained and not use_sudo and not asked_sudo and interactive:
            what = [f"端口 {p}" for p in res.ports_unexplained] + [
                f"{c} 上有 CAN 接收者" for c in res.can_unexplained
            ]
            print(f"\n发现无法归属的占用（{', '.join(what)}），可能是其他用户的进程。")
            asked_sudo = True
            if ask("用 sudo 检查？[Y/n]: ", "y") in ("y", "yes"):
                use_sudo = True
                continue
        groups = group({p: r for p, r in res.holders.items() if p not in (ignore_pids or set())})
        down = [] if ports_only else down_ifaces(hw_config)
        if not groups and not unexplained and not down:
            print("硬件空闲：CAN / 相机 / RPC 端口都没有被占用，CAN 口全部 UP。")
            return True
        if groups:
            print("以下程序占用着机械臂相关资源：")
            describe(groups)
        if unexplained and not groups:
            print(f"\n仍有无法归属的占用：端口 {res.ports_unexplained} CAN {res.can_unexplained}")
        if down:
            print(f"\nCAN 口未就绪: {' '.join(down)}")
        if not interactive:
            return False
        if groups:
            picked = choose(groups)
            if not picked:
                print("未停止任何程序。")
                return False
            for g in picked:
                print(f"\n停止 [{groups.index(g) + 1}] pid {g.owner} ...")
                stop(g)
            time.sleep(0.5)
            continue
        if down:
            if ask("重置 CAN 口？（需要 sudo 密码）[Y/n]: ", "y") not in ("y", "yes"):
                return False
            subprocess.run(
                ["sh", str(PROJECT_ROOT / "third_party/i2rt/scripts/reset_all_can.sh")], check=False
            )
            time.sleep(0.5)
            continue
        return False
    return False


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hw-config", type=Path, default=DEFAULT_HW_CONFIG)
    p.add_argument("--check", action="store_true", help="only list, never stop (exit 1 if busy)")
    p.add_argument("--sudo", action="store_true", help="inspect other users' processes via sudo")
    p.add_argument("--scan-json", action="store_true", help=argparse.SUPPRESS)  # used via sudo
    args = p.parse_args()
    if args.scan_json:
        res = scan(args.hw_config)
        print(
            json.dumps(
                {
                    "holders": res.holders,
                    "ports_unexplained": res.ports_unexplained,
                    "can_unexplained": res.can_unexplained,
                }
            )
        )
        return
    sys.exit(0 if free(args.hw_config, interactive=not args.check, use_sudo=args.sudo) else 1)


if __name__ == "__main__":
    main()
