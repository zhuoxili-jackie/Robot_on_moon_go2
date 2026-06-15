# Robot on Moon — Go2 运控训练（TRAINING-Go2）

宇树 **Go2** 四足在 **MuJoCo + Gymnasium + Stable-Baselines3（PPO）** 上的运控训练包。
本仓库是「Robot on Moon」项目的 Go2 训练子包，总目标是让 Go2 在**月面地形**行走，分四个阶段推进：

> **执行顺序（杜老师调整）：Phase 1 ✅ → Phase 3 ✅ → Phase 2（当前）→ Phase 4。** 阶段编号不变，只换顺序。

- **Phase 1 — 裸机平地行走** ✅ 已完成（仓库 v1.4：定型 G/J/N 三套配置）：不带机械臂的 Go2 在平地稳定全向行走。
- **Phase 3 — Go2 + Z1** ✅ 已完成（仓库 v3.1）：把 Z1 机械臂作为**配重**焊到 go2 base、平地重训行走（本步不控臂）。带臂模型 `go2_z1.xml`（kp45），正式 run `go2_z1_flat_final_3p5M` eval **5/5**，全面优于裸机 N。
- **Phase 2 — 月面迁移**（**当前**）：在带 Z1 配重的 `go2_z1.xml` 上接入月球高度场地形，验证迁移 / 重整（规划见根目录 `CONTINUATION_PROMPT_robot_on_moon.md` §三）。
- **Phase 4 — 键盘实时运控**：在 MuJoCo viewer 中接入键盘事件，仿真运行时实时调速 / 转向，人机交互驾驶。

> 不使用 Isaac；团队自研的轻量 MuJoCo 训练场。默认简体中文。

## 快速开始

详细安装 / 训练 / 回放 / 验收命令见 **[`TRAINING.md`](TRAINING.md)**。最常用：

```powershell
python -m pip install -r requirements.txt
python eval_policy.py --run go2_z1_flat_final_3p5M --xml go2_z1_flat_scene.xml --render                # 验收带臂正式版（Phase 3，5/5）
python eval_policy.py --run go2_gN_tc55_3p5M --render                                                 # 验收裸机 N（Phase 1）
python train_ppo.py --total-timesteps 3500000 --num-envs 4 --run-name my_run --checkpoint-freq 250000  # 自己训一版（默认 3.5M）
```

## 验收标准与结果

验收**不靠肉眼、用数值阈值**：`eval_policy.py` 跑 5 个固定指令的确定性回放，量化速度跟踪 / 占空比 /
触地频率 / 力矩&动作饱和 / 腾空比 / roll-pitch / 站高，对照阈值逐项 PASS/FAIL。
**完整阈值定义 + 每一代的验收结果记录在 [`RESULTS.md`](RESULTS.md)。**

## 目录

| 路径 | 说明 |
|---|---|
| `go2_env.py` | Gymnasium 环境 `Go2WalkEnv`（obs 54 / act 12 / 行走 reward） |
| `train_ppo.py` | PPO 训练入口（SB3，含 `--checkpoint-freq`） |
| `play_policy.py` | viewer 回放策略 |
| `eval_policy.py` | 量化验收（指标 + 阈值 + 步态图 / gif） |
| `smoke_test.py` | env 随机动作冒烟测试 |
| `go2.xml` / `go2_flat_scene.xml` / `go2_assets/` | 裸机 Go2 模型（PD 执行器 kp35）+ 平地场景 + 网格（Phase 1，env 默认场景） |
| `go2_z1.xml` / `go2_z1_flat_scene.xml` / `z1_assets/` | 带 Z1 配重的 Go2（kp45）+ 场景 + Z1 网格（Phase 3）；与 go2.xml 独立 |
| `TRAINING.md` | 详细运行指南 |
| `RESULTS.md` | 验收标准 + 各代结果 |