# Robot on Moon — Go2 运控训练（TRAINING-Go2）

宇树 **Go2** 四足在 **MuJoCo + Gymnasium + Stable-Baselines3（PPO）** 上的运控训练包。
本仓库是「Robot on Moon」项目的 Go2 训练子包，总目标是让 Go2 在**月面地形**行走，分四个阶段推进：

> **执行顺序（杜老师调整）：Phase 1 ✅ → Phase 3 ✅ → Phase 2 ✅ → Phase 4（当前）。** 阶段编号不变，只换顺序。

- **Phase 1 — 裸机平地行走** ✅ 已完成（仓库 v1.4：定型 G/J/N 三套配置）：不带机械臂的 Go2 在平地稳定全向行走。
- **Phase 3 — Go2 + Z1** ✅ 已完成（仓库 v3.1）：把 Z1 机械臂作为**配重**焊到 go2 base、平地重训行走（本步不控臂）。带臂模型 `go2_z1.xml`（kp45），正式 run `go2_z1_flat_final_3p5M` eval **5/5**，全面优于裸机 N。
- **Phase 2 — 月面迁移** ✅ 已完成（仓库 v2.1）：带 Z1 配重的 Go2 迁到**程序化月面高度场**（平台 + 起伏 + 大坑 + 山丘）。正式 run `go2_lunar_3p5M`（`Go2LunarEnv` 地形相对高度、SubprocVecEnv 多核训），eval **5/6、6/6 不摔**：爬坡 / 穿大坑 / 过起伏 / 转向全 PASS，仅最深 0.40m 坑安全停沿不下。
- **Phase 4 — 键盘实时运控**（**当前**）：在 MuJoCo viewer 中接入键盘事件，仿真运行时实时调速 / 转向，人机交互驾驶。

> 不使用 Isaac；团队自研的轻量 MuJoCo 训练场。默认简体中文。

## 快速开始

详细安装 / 训练 / 回放 / 验收命令见 **[`TRAINING.md`](TRAINING.md)**。最常用：

```powershell
python -m pip install -r requirements.txt
python play_policy.py --lunar --model runs/go2_lunar_3p5M/ppo_go2_final.zip --vecnormalize runs/go2_lunar_3p5M/vecnormalize.pkl  # 看月面 walker（Phase 2）
python eval_policy.py --lunar --run go2_lunar_3p5M --render                                           # 验收月面正式版（Phase 2，5/6、6/6 不摔）
python eval_policy.py --run go2_z1_flat_final_3p5M --xml go2_z1_flat_scene.xml --render               # 验收带臂正式版（Phase 3，5/5）
# 训练（★多核：SubprocVecEnv 12 env 比默认 4 env DummyVecEnv 快 ~4.7×，月面 3.5M ~20min）
python train_ppo.py --lunar --subproc --num-envs 12 --torch-threads 4 --total-timesteps 3500000 --run-name my_lunar --checkpoint-freq 250000
```

## 验收标准与结果

验收**不靠肉眼、用数值阈值**：`eval_policy.py` 跑 5 个固定指令的确定性回放，量化速度跟踪 / 占空比 /
触地频率 / 力矩&动作饱和 / 腾空比 / roll-pitch / 站高，对照阈值逐项 PASS/FAIL。
**完整阈值定义 + 每一代的验收结果记录在 [`RESULTS.md`](RESULTS.md)。**

## 目录

| 路径 | 说明 |
|---|---|
| `go2_env.py` | Gymnasium 环境 `Go2WalkEnv`（obs 54 / act 12 / 行走 reward）；含 `_ground_height()` 钩子 |
| `go2_lunar_env.py` | `Go2LunarEnv(Go2WalkEnv)`：月面**地形相对高度**（mj_ray 查 hfield 地表）+ 跨地形 spawn（Phase 2） |
| `train_ppo.py` | PPO 训练入口（SB3）；`--lunar` 月面、**`--subproc` 多核**、`--torch-threads`、warm-start |
| `play_policy.py` | viewer 回放策略（`--lunar` 月面） |
| `eval_policy.py` | 量化验收（指标 + 阈值 + 步态图 / gif）；`--lunar` 用月面阈值 + 跨坑/坡回合 |
| `smoke_test.py` | env 随机动作冒烟测试 |
| `go2.xml` / `go2_flat_scene.xml` / `go2_assets/` | 裸机 Go2 模型（PD 执行器 kp35）+ 平地场景 + 网格（Phase 1，env 默认场景） |
| `go2_z1.xml` / `go2_z1_flat_scene.xml` / `z1_assets/` | 带 Z1 配重的 Go2（kp45）+ 场景 + Z1 网格（Phase 3）；与 go2.xml 独立 |
| `go2_lunar_scene.xml` / `lunar_assets/` | 月面 hfield 场景（带 Z1）+ 程序化地形生成器与资产（Phase 2） |
| `TRAINING.md` | 详细运行指南 |
| `RESULTS.md` | 验收标准 + 各代结果 |