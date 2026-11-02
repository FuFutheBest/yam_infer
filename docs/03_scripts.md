# 03 脚本与命令

> 机器人上的命令都在项目根目录（`~/fufu/yam_infer`）执行。

## 1. 总览

```mermaid
flowchart LR
    S["setup_robot.sh<br/>一次性装环境 + 自检"]
    S --> D["yam_infer.run --dry-run<br/>离线演练"]
    D --> R["run_robot.sh<br/>真机（模型见 models.yaml）"]
    F["free_hw.sh<br/>谁占着硬件 → 问要不要停"] -. 自动调用 .-> R
    R --> V["tools.review<br/>复盘"]
    R -. 异常后手臂不在位 .-> H["home_arms.sh<br/>任何状态下归位"]
    H -. 需要接管时调用 .-> F
```

| 命令 | 在哪跑 | 什么时候用 | 动硬件？ |
|---|---|---|---|
| `bash scripts/setup_robot.sh [--dev]` | 机器人 | 第一次 / 依赖变了 / 重建 `.venv` | 否 |
| `.venv/bin/python scripts/check_env.py` | 机器人 | 环境自检（setup 最后会自动跑） | 否 |
| `.venv/bin/python -m pytest -q` | 两边 | 改代码后 | 否 |
| `.venv/bin/python -m yam_infer.run … --dry-run <录制>` | 两边 | 不碰硬件跑一遍全流程 | 否 |
| `bash scripts/run_robot.sh [额外参数]` | 机器人 | 真机运行；模型、任务、增量/绝对都在 `configs/models.yaml` | **是** |
| `.venv/bin/python -m yam_infer.tools.review runs_infer/<目录> --video` | 两边 | 复盘 | 否 |
| `bash scripts/free_hw.sh [--check] [--sudo]` | 机器人 | 开不了机械臂时：列出占用 CAN / 相机 / RPC 端口的程序并问要不要停；CAN 口没 UP 就问要不要重置（`run_robot.sh` 自动调用） | 停程序可能让手臂断力 |
| `bash scripts/home_arms.sh` | 机器人 | 不管 worker 什么状态，把两只手臂慢速归位再断电 | **是** |
| `bash scripts/vendor_diff.sh <SI 目录>` | 用户电脑 | 检查 SI 上游有没有更新 worker / i2rt | 否 |
| `python -m yam_infer.tools.export IN.pt OUT.pt` | 任意 | 训练 ckpt → 部署模型 | 否 |
| `python -m yam_infer.policy.server --ckpt … --host <IP>` | GPU 机 | 机器人没 GPU 时 | 否 |

## 2. `scripts/setup_robot.sh`

```mermaid
flowchart LR
    A["uv venv .venv<br/>/usr/bin/python3.10"] --> B["torch 2.7.1 + torchvision 0.22.1<br/>cu128 源"] --> C["requirements.txt<br/>阿里云镜像"] --> P["写 .pth：<br/>third_party/i2rt 可导入"] --> D["check_env.py 自检"]
```

| 注意 | 说明 |
|---|---|
| Python 3.10 | 与 i2rt 驱动验证过的版本一致 |
| cu128 | RTX 50 系（5090）只能用 CUDA 12.8 的 torch |
| numpy 2.2.6 | i2rt 固定版本 |
| `--dev` | 额外装 pytest、ruff |
| `PIP_MIRROR=…` | 换 PyPI 镜像 |

`check_env.py` 检查：torch + GPU、numpy/cv2、pyrealsense2/portal、i2rt 依赖、i2rt SDK 能生成 YAM+crank_4310 的 MuJoCo 模型、`hardware.yaml` 端口与相机、`models/*.pt`、worker 模块可导入。最后一行 `ALL OK` 才算通过。

## 3. `scripts/run_robot.sh [额外参数]`

```mermaid
flowchart TB
    A["读 robot.env<br/>HW_CONFIG、可选 CKPT"] --> A2{".venv 存在？"}
    A2 -->|否| X0["报错退出"]
    A2 -->|是| B["free_hw：列出占用者 → 问要不要停<br/>CAN 口没 UP → 问要不要重置"]
    B -->|仍被占用 / 你选了不停| X["退出，不启动"]
    B -->|硬件空闲| F["exec .venv/bin/python -m yam_infer.run<br/>--task（+ --ckpt）--hw-config 额外参数<br/>run.py 按 models.yaml 选模型并检查"]
```

| 变量（`robot.env`） | 默认 |
|---|---|
| `CKPT` | 不设 = 用 `configs/models.yaml` 的模型；临时换同类模型：`CKPT=models/act_v2_delta_s15k.pt bash scripts/run_robot.sh`（仍按 models.yaml 校验格式 / 任务 / 增量绝对） |
| `HW_CONFIG` | `configs/hardware.yaml` |
| `YAM_LEFT_CAM_SERIAL` / `YAM_RIGHT_CAM_SERIAL` | 不设则用 `hardware.yaml` 的 `cameras:` |

### 运行时按键（输入后回车）

| 按键 | 作用 |
|---|---|
| 回车 | 开始 / 暂停 / 继续（暂停时原地保持） |
| `r` | 结束本回合（保存记录）→ 回起始姿态 → 等下一回合 |
| `q` | 回起始姿态 → 原路回休息姿态 → 断力 |
| Ctrl+C（或 SIGTERM） | 中断，同样先归位再断电；归位过程中再按 Ctrl+C 会被忽略（防止半路断电掉臂） |

## 4. `yam_infer.run` 全部参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--models-config` | `configs/models.yaml` | 模型及其解读方式 |
| `--ckpt` | `models.yaml` 的 `ckpt` | 临时换模型文件 |
| `--task` | `models.yaml` 的 `task` | 临时换任务（旧多任务模型里选另一个） |
| `--hw-config` | `configs/hardware.yaml` | 从臂 + 相机配置 |
| `--left-serial` / `--right-serial` | 配置文件 | 相机序列号覆盖 |
| `--remote host:port` | 关 | 用远程推理服务 |
| `--device` | `auto` | `cuda` / `cpu` |
| `--no-jpeg-roundtrip` | 关 | 跳过 JPEG 编解码（不建议） |
| `--goto-start` | `task_mean` | 每回合前移到示范平均起始姿态；`none` 不动 |
| `--park` | `home` | 退出时：`home` 回休息姿态再断电；`start` 只回起始姿态；`none` 直接断电（**会掉**） |
| `--exec-mode` | 读 `run.yaml` `execution.mode`（chunk） | `chunk` 承诺执行 / `ensemble` 时间集成 |
| `--exec-horizon` | 读 `execution.horizon`（30） | chunk 模式每个计划执行多少步 = 多久重新看图（30 步 = 1 s），1 到模型的 chunk 长度（act_v2 50、act_ref 100） |
| `--blend-s` | 读 `execution.blend_s`（0.1） | chunk 模式换计划时的交叉淡化时长 |
| `--ensemble-m` | 读 `execution.ensemble_m`（0.01） | ensemble 模式权重系数 |
| `--margin` | 0.15 | 允许超出示范范围的弧度 |
| `--max-joint-speed` | 1.5 | rad/s；**新模型第一次上机建议 0.8** |
| `--freeze-left` / `--no-freeze-left` | 读 `run.yaml` `control.freeze_left`（false） | 冻结左臂：保持回合开始姿态，不跟网络输出（`--move-left` = 旧别名，等于 `--no-freeze-left`） |
| `--freeze-right` / `--no-freeze-right` | 读 `control.freeze_right`（false） | 同上，右臂 |
| `--run-config` | `configs/run.yaml` | 运行参数文件 |
| `--max-seconds` | 读 `run.yaml` 的 `episode.max_seconds`（40） | 单回合最长时间；命令行给了就以命令行为准 |
| `--record-dir` | `runs_infer` | 记录目录 |
| `--dry-run <录制目录>` | 关 | 离线演练：录制画面 + 模拟臂 |
| `--fake-arms` | 关 | 配合 `--dry-run`：模拟臂跑在**真实的 worker 进程**里，走完整 RPC/心跳/看门狗路径 |

## 5. 离线演练

```bash
# 最快：模拟臂在主进程里
.venv/bin/python -m yam_infer.run \
    --dry-run sample_episodes/put_banana_in_bowl/episode_0019 --record-dir runs_infer_dryrun
# 更接近真机：真实 worker 进程 + 模拟臂（端口 1234/1235 必须空闲，采集 WebUI 开着时会被拒绝）
… --fake-arms
```

| 检查 | 期望 |
|---|---|
| 启动 → 对齐 → 回车运行 → r / q | 全流程不报错 |
| `latency` | 机器人 GPU 约 15 ms；用户电脑 CPU 约 150–250 ms |
| 记录目录 | `data/*.mcap、policy.npz、command.npz、events.jsonl、episode_stats.json、meta.json`；`episode_stats.json` 里 `dropped_camera_frames` = 0、`writer_error` = null |

![离线演练复盘](img/dryrun_review.png)

## 6. 复盘

```bash
.venv/bin/python -m yam_infer.tools.review runs_infer/<目录> --video
```

| 输出 | 内容 |
|---|---|
| 终端 JSON | 结束原因、策略频率、延迟 p50/p95、安全层修改比例、右臂跟踪误差；新格式另有丢帧数、写盘错误、图像年龄、各相机帧数 |
| `review.png` | 14 个关节：红 = 模型目标，绿虚线 = 实际下发，蓝 = 从臂实际；延迟、推理间隔、图像年龄 |
| `review.mp4` | 新格式：两路相机**按真实速度**播放（30 fps），被模型用到的那一帧加绿框，下方是当时最新计划的右夹爪曲线；旧格式：模型输入画面 |

```bash
# 离线重跑模型，验证录制能复现网络输出（机器人上用 --device cuda 和现场一致）
.venv/bin/python -m yam_infer.tools.replay_policy runs_infer/<目录> --ckpt models/act_v2_all.pt
# 推理录制本身也是 NEDF3，可以直接当离线演练的输入
.venv/bin/python -m yam_infer.run --dry-run runs_infer/<目录>
```

| 目录 | 含义 |
|---|---|
| `runs_infer/<时间>_<任务>/` | 完整回合 |
| `runs_infer/quarantine/` | 不完整的回合（缺数据、写盘出错、程序中途崩溃），原因见 `quarantine_reason.json` |

| 现象 | 看哪里 | 可能原因 / 处理 |
|---|---|---|
| 动作卡顿 | `latency_ms_p95` > 33 | 推理太慢 → `--remote` |
| 动作被截断 | `frac_commands_changed_by_safety_or_freeze.right_arm` 高 | 模型越界或太快 → 调 `--margin` / 速度 |
| 实际跟不上目标 | review.png 蓝线落后红线 | 正常约 130 ms；更大看 worker 日志 |
| 停住不动 | review.png 右臂曲线变平；`end_reason` = max_seconds | 调大 `--exec-horizon`；或模型能力 |
| 抓错物体 | review.mp4 | 补数据再训练 |

## 7. `scripts/free_hw.sh`：谁占着硬件

```mermaid
flowchart TB
    S["扫描 /proc（不靠程序名，看真实占用）"] --> C1["CAN socket<br/>（协议名 CAN_RAW…）"]
    S --> C2["RPC 端口 1234–1237<br/>监听者 + 连着它的客户端"]
    S --> C3["RealSense 相机<br/>/dev/video*、/dev/bus/usb（Intel）"]
    S --> C4["CAN 口状态<br/>hardware.yaml 里的 can0–3"]
    C1 & C2 & C3 --> G["按「所属程序」分组<br/>向上找到第一个不是 shell/终端的祖先<br/>（例：4 个 SI worker + 2 个相机进程 → SI WebUI）"]
    G --> Q{"停止哪些？ a / 1,2 / n"}
    Q -->|选中| K["SIGINT（=Ctrl+C）→ SIGTERM → SIGKILL<br/>别人的进程用 sudo kill"]
    K --> S
    C4 -->|有没 UP 的| R{"重置 CAN 口？"} -->|是| RS["reset_all_can.sh（sudo 问密码）"] --> S
    Q -->|n| X["exit 1"]
    G -->|什么都没有| OK["硬件空闲 exit 0"]
```

| 选项 | 作用 |
|---|---|
| （无） | 列出 → 问 → 停 → 重新扫描，直到空闲或你选择不停 |
| `--check` | 只列出，不停任何东西；忙则 exit 1 |
| `--sudo` | 用 sudo 扫描其他用户的进程（普通扫描看不到别人进程的文件句柄）。发现「端口被占但找不到是谁」时会自动问要不要 |
| `--hw-config` | 读端口和 CAN 口名 |

| 输出标记 | 含义 |
|---|---|
| `[本项目]` | `yam_infer.run` 或本项目 worker |
| `! 它在控制机械臂` | 有 CAN socket：停掉它电机就断力，**手臂会落下** → 想先归位用 `home_arms.sh` |
| `连着 RPC 端口` | 它是控制者（如 WebUI 的协调器、推理程序） |

## 8. `scripts/home_arms.sh`：任何状态下归位

```mermaid
flowchart TB
    T["目标：configs/rest_pose.yaml（休息姿态）<br/>或 --target zero"] --> Y{"确认移动？"}
    Y --> R0["本项目 yam_infer.run 还在跑？→ SIGINT，它自己归位退出"]
    R0 --> R1["[1] 端口上有 worker 且没人在控制<br/>→ 清 FAULT → RPC 慢速归位"]
    R1 -->|两只都到位| E
    R1 -->|"被别人控制 / worker 卡死 / 没有 worker"| F["[2] free_hw：列出占用者，问你停不停"]
    F --> W["没有 worker 的臂：启动本项目 worker<br/>（先原地保持）→ 慢速归位"]
    W --> E["本项目 worker 停止 → 在休息姿态断电<br/>别人的 worker（如 SI）保留，它继续保持"]
```

| 选项 | 作用 |
|---|---|
| `--target rest`（默认）/ `zero` | `configs/rest_pose.yaml` / 关节全 0、夹爪张开 |
| `--rest-pose <yaml>` | 换一个休息姿态文件 |
| `--speed 0.4` | 归位峰值速度 rad/s |
| `--keep-on` | 归位后保持通电，Ctrl+C 才断电 |
| `--yes` | 不问确认 |
| `--fake` | 模拟 worker（测试用） |

> 安全细节：只在 worker 没有被别人控制时才接管（先 `get_status`，不会发出会被拒绝的请求——worker 收到被拒绝的请求会直接退出、手臂落下）。

## 9. 远程推理（可选）

```bash
# GPU 机器
python -m yam_infer.policy.server --ckpt models/act_v2_all.pt --host <局域网 IP> --port 5555
# 机器人
bash scripts/run_robot.sh --remote <GPU 机 IP>:5555
```

## 10. pi0.5 模型（`format: pi05`，`pi05` 分支）

pi0.5 有 3.3B 参数，用 JAX 写成，依赖由 openpi 固定（jax 0.5.3、numpy<2 等），和本项目的环境（numpy 2.2.6）装不到一起。所以网络跑在一个独立的**策略服务**进程里，`yam_infer.run` 只是客户端。这也是 openpi 官方的部署方式。

服务端代码在本项目 `yam_infer/pi05_server/`（从 yam_train `pi05` 分支搬来，见 VENDORED.md），**运行时不需要 yam_train**，只需要一个 openpi 环境（openpi @215abfb + uv `.venv`）和导出的模型目录。

```mermaid
flowchart LR
    RR["run_robot.sh"] --> Y
    subgraph R["机器人"]
        Y["yam_infer.run（本项目 .venv）<br/>models.yaml: format pi05, server 127.0.0.1:8000, model_dir"]
        S["策略服务（openpi .venv，GPU）<br/>python -m yam_infer.pi05_server --model models/pi05_yam"]
    end
    Y -->|"端口无人监听 → 自动启动 / 退出时停止"| S
    Y <-->|"websocket（msgpack）<br/>2×224×224 图 + 状态 + 指令 → 50×14 绝对目标"| S
```

| 步骤 | 命令 / 配置 |
|---|---|
| 1. `configs/models.yaml` | pi05 块：`server: 127.0.0.1:8000`、`model_dir: models/pi05_yam`、`format: pi05`、`task: …`、`action_mode: delta` |
| 2. `configs/run.yaml` | `mode: chunk`、`horizon: 25`、`blend_s: 0.1`、`freeze_left: true` |
| 3. 运行 | `bash scripts/run_robot.sh`：自动启动服务 → 等待加载 + 编译（约 1–2 分钟，每 10 s 打印进度）→ 对齐 → 回合循环 → 退出时停止服务。也可以临时用 `--task` 覆盖 models.yaml |

自动启动的细节（`yam_infer/policy/pi05_launch.py`）：

| 项 | 行为 |
|---|---|
| 何时启动 | 有 `model_dir`，且 `server` 端口没有进程监听。`model_dir` 只能配本机地址（127.0.0.1 / localhost） |
| 端口已有服务 | 直接使用，但要求它服务的就是 `model_dir`（比较 config / experiment / step / repo_id），否则报错退出 |
| openpi 的 python | `OPENPI_PY` → `$OPENPI_DIR/.venv/bin/python` → 默认 `<项目>/../openpi`、`~/fufu/astra_inference_train/openpi`（机器人上现有的 openpi @215abfb 环境）。可在 `robot.env` 里 `export OPENPI_DIR=...` |
| 显存 | 默认 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`（JAX 默认会占 75%，显卡上可能有别人的进程）；在 `robot.env` 里 export 可改 |
| 日志 | `runs_infer/pi05_server.log`（每次覆盖）；服务启动失败时把日志末尾打印出来 |
| 停止 | 正常退出、`q`、Ctrl+C、报错都会停止（先 SIGTERM，15 s 后 SIGKILL）。服务在独立进程组，Ctrl+C 不会直接打断它 |
| 进程被 `kill -9` | 服务会留下；下次 `run_robot.sh` 发现端口有服务、且模型一致，就直接复用 |
| 不用自动启动 | 去掉 `model_dir`，自己在任何 GPU 机启动：`PYTHONPATH=<本项目> <openpi python> -m yam_infer.pi05_server --model models/pi05_yam --host 0.0.0.0 --port 8000`，`server:` 写那台机的地址 |

- 一个模型同时会两个任务，`task` 决定发给模型的英文指令。
- 起始姿态、动作范围（安全过滤用）、fps 和相机列表都由服务在连接时发过来。
- 图像预处理和训练完全一致：先做 JPEG q90 往返（用录制器的编码器），再用 `resize_with_pad` 缩放到 224×224（PIL 双线性，等比缩放后上下补黑边）。
- 依赖：`websockets`、`pillow`（已写入 requirements.txt）。
- 已验证：在用户电脑上用 `--dry-run` 回放录制数据，经 SSH 隧道连到服务器上的策略服务（训练 1000 步的模型，CPU 推理），整条链路正常。
- **未验证**：机器人 RTX 5090（Blackwell）上能否直接运行 JAX 0.5.3。如果报 `ptxas too old`，可以换新版 `nvidia-cuda-nvcc-cu12`（≥12.8），或者改把服务放到训练服务器上跑，用 SSH 隧道连接。
