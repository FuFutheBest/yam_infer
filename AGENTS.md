# AGENTS.md — yam_infer

- 从 `~/Dev/MVIG/opus_inference_train/opus/{infer,model,tools}` 拆出的**独立推理框架**；训练、数据转换仍在 opus_inference_train。
- `yam_infer/model/act.py` 必须与训练端 `opus/model/act.py` 保持一致（改网络结构时两边同步）。
- 机器人：`team1@10.128.1.165`，SuperInference 在 `~/桌面/Project_MVIG/SuperInference-feat-i2rt_NEDF3`（只借用其 follower worker，不修改）。
- 安全不变量：worker 150 ms 指令租约 → 必须有心跳；退出前归位到休息姿态；worker 随主程序退出；夹爪挤压上限。
- 文档风格：mermaid + 表格，少文字。
