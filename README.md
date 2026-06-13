# Robot on Moon — Go2 运控训练（TRAINING-Go2）

宇树 **Go2** 四足在 **MuJoCo + Gymnasium + Stable-Baselines3（PPO）** 上的运控训练包。
本仓库是「Robot on Moon」项目的 Go2 训练子包，总目标是让 Go2 在**月面地形**行走，分三个阶段推进：

- **Phase 1 — 裸机平地行走**（当前）：不带机械臂的 Go2 在平地稳定全向行走。
- **Phase 2 — 月面迁移**：接入月球高度场地形，验证迁移 / 重整。
- **Phase 3 — Go2 + Z1**：装上 Z1 机械臂做整机 loco-manipulation。

> 不使用 Isaac；团队自研的轻量 MuJoCo 训练场。默认简体中文。

## 快速开始

详细安装 / 训练 / 回放 / 验收命令见 **[`TRAINING.md`](TRAINING.md)**。最常用：

```powershell
python -m pip install -r requirements.txt
python train_ppo.py --total-timesteps 5000000 --num-envs 4 --run-name go2_baseline_5M --checkpoint-freq 250000
python eval_policy.py --run go2_baseline_5M --render
```

## 验收标准与结果

验收**不靠肉眼、用数值阈值**：`eval_policy.py` 跑 5 个固定指令的确定性回放，量化速度跟踪 / 占空比 /
触地频率 / 力矩&动作饱和 / 腾空比 / roll-pitch / 站高，对照阈值逐项 PASS/FAIL。
**完整阈值定义 + 每一代的验收结果记录在 [`RESULTS.md`](RESULTS.md)。**

## 目录

| 路径 | 说明 |
|---|---|
| `go2_env.py` | Gymnasium 环境 `Go2WalkEnv`（obs 52 / act 12 / 行走 reward） |
| `train_ppo.py` | PPO 训练入口（SB3，含 `--checkpoint-freq`） |
| `play_policy.py` | viewer 回放策略 |
| `eval_policy.py` | 量化验收（指标 + 阈值 + 步态图 / gif） |
| `smoke_test.py` | env 随机动作冒烟测试 |
| `go2.xml` / `go2_flat_scene.xml` / `go2_assets/` | Go2 模型（PD 执行器）+ 平地场景 + 网格 |
| `TRAINING.md` | 详细运行指南 |
| `RESULTS.md` | 验收标准 + 各代结果 |