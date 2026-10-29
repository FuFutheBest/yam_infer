# AGENTS.md

## 任务目标

为 i2rt yam 双臂主从遥操作机器人，从完成推理部分代码。 用户电脑负责开发，最终在**机器人电脑**运行（用户电脑无硬件，不测试真机）。

## 机器区分

见 [SEVERS.md](SEVERS.md)。

## 分析依据

- 数据采集项目：`~/Dev/MVIG/SuperInference-feat-i2rt_NEDF3/`（采集端，不能在用户机器运行）
- 网络训练项目：`~/Dev/MVIG/yam_train/`
- 用户的 prompt
- **不要参考** `~/Dev/MVIG` 下和服务器上的其他尝试（dp / pi0 / pi0.5 / act 等目录），独立分析决定方案。

## 工作约定

- 用户不懂具身/神经网络 → 需要我独立探索、自主设计架构，并在文档中解释用到的知识。
- 文档风格：mermaid 图 + 表格优先，避免大段 README 式文字。
- 训练后网络可能还要调整，后续在同一会话继续迭代；保持代码、配置、日志可复现。
- 不修改源项目 `SuperInference-feat-i2rt_NEDF3` 中的文件（只读引用其代码与数据）。

## 当前状态

| 项 | 值 |
|---|---|
| 独立性 | 自包含：从臂 worker + i2rt SDK 已搬入（`yam_infer/hw/follower/`、`third_party/i2rt/`，清单见 `VENDORED.md`），运行时不再需要 SI |
| 机器人部署 | `team1@10.128.1.165:~/fufu/yam_infer`（`check_env` ALL OK、15 测试通过、GPU 离线演练通过）；**独立版真机待测** |
| 旧部署 | `~/fufu/opus_inference_train`（依赖 SI worker，真机已跑成功），保留作回退 |
| 开发流程 | 见 `docs/06_development.md` |
| 约束 | 不能与采集 WebUI 同时运行（争用 CAN/相机/端口 1234–1235）；真机测试须有人在旁 |
