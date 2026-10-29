# Vendored third-party code

本项目不再依赖 SuperInference（SI）运行。下面这些代码从 SI 复制而来，**改动时要记在这里**。

来源：`SuperInference-feat-i2rt_NEDF3` @ `0dbfa5dc37a4bd2067215284429d0415d02d53cd`（机器人上跑通真机的同一版本；机器人与用户电脑这两个目录均无本地修改）。

| 本项目 | 来源 | 本地改动 |
|---|---|---|
| `yam_infer/hw/follower/runtime.py` | `devices/robots/i2rt_teleop/runtime.py` | 无（逐字节一致） |
| `yam_infer/hw/follower/config.py` | `devices/robots/i2rt_teleop/config.py` | 无（`REPO_ROOT` 仍为 `parents[3]` = 本项目根目录） |
| `yam_infer/hw/follower/endpoints.py` | `devices/robots/i2rt_teleop/endpoints.py` | 只保留从臂；只支持一体化 `i2rt_can` 夹爪（去掉 LKTech 串口夹爪）；新增 `FakeFollowerEndpoint` |
| `yam_infer/hw/follower/worker.py` | `devices/robots/i2rt_teleop/worker.py` | 只保留 follower；用标准 logging 替代 SI 的 loguru（不再在 cwd 写 `logs/`）；新增 `--fake`。RPC 接口与关闭顺序不变 |
| `configs/hardware.yaml` | `configs/i2rt_d405_yam_teleop.yaml` | 新增 `cameras:`（原来读 SI WebUI 的 `.superinference/local/i2rt_hardware_bindings.json`） |
| `third_party/i2rt/` | `third_party/i2rt/`（i2rt 1.1.2，MIT） | 裁剪：删掉 `tests/`、`flow_base/`、`docs/`、`examples/`、`devices/`，以及本机不用的机械臂/夹爪模型（`big_yam`、`yam_pro`、`yam_ultra`、`flexible_4310`、`linear_3507`、`linear_4310`、`station`）。代码未改 |
| `third_party/i2rt/scripts/reset_all_can.sh` | 同名 | 无 |
| `yam_infer/data/nedf3_schemas/*.desc` | SI 采集文件里嵌入的 protobuf 定义（`nmx.msg.Metadata` / `ArmState` / `HandState` / `RGBD`），用 `scripts/extract_nedf3_schemas.py` 提取 | 无（原样字节） |
| `yam_infer/data/nedf3_writer.py` 的话题、字段用法 | `consumer/io/nedf3_i2rt_contract.py`、`async_data_saver_i2rt_NEDF3.py` | 自己实现（不用 dm3_core）；`header.seq` / `stamp` 填值；夹爪 100 Hz；无主臂话题 |
| `tests/test_follower_runtime.py` | `tests/devices/test_i2rt_teleop_watchdog.py` | 只移植 runtime 相关用例 |

## 不再需要的 SI 东西

| 原来 | 现在 |
|---|---|
| SI 的 `.venv/bin/python` 跑 worker | 本项目 `.venv` 跑 worker（同为 Python 3.10 + numpy 2.2.6） |
| `--si-root` / `robot.env` 里的 `SI_ROOT` | 删除 |
| `SI_I2RT_YAM_GRIPPER_TYPE=crank_4310` | 删除（worker 从未读取它；夹爪类型写在 `configs/hardware.yaml`） |
| `SI_I2RT_{LEFT,RIGHT}_RS_SERIAL` | `YAM_{LEFT,RIGHT}_CAM_SERIAL` 或 `configs/hardware.yaml` |

## 与 SI 共存

采集 WebUI 的 worker 用同样的 CAN 口和 RPC 端口（1234/1235）。`run_robot.sh` 启动前调 `free_hw` 列出它并问是否停止；`FollowerArms` 发现端口被占用会拒绝启动（绝不连到别人的 worker）；`home_arms` 只在 worker 无人控制时才通过 RPC 接管。

> 注意：portal 3.7.3 的 `Server` 默认把 RPC 方法抛出的异常再抛出，worker 进程随之退出（手臂断力）。所以任何会被 worker 拒绝的请求（会话不匹配、帧过期、序号倒退）都等于让手臂落下——调用方必须先查状态再发请求。

## 重新同步 SI 的更新

```bash
bash scripts/vendor_diff.sh ~/Dev/MVIG/SuperInference-feat-i2rt_NEDF3   # 列出与上游的差异
```
