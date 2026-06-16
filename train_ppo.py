"""PPO training entry point for Go2 walking.

Ported from TRAINING-Aliengo/train_ppo.py (same PPO hyperparameters); only the env
and output names differ. Run from this directory so outputs land in ./runs/:

    python train_ppo.py --total-timesteps 5000000 --num-envs 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from go2_env import Go2WalkEnv
from go2_lunar_env import Go2LunarEnv


def make_env(xml_path: Path | None, seed: int, rank: int, lunar: bool = False):
    def _init():
        env = Go2LunarEnv(xml_path=xml_path) if lunar else Go2WalkEnv(xml_path=xml_path)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Go2 walking with PPO.")
    parser.add_argument("--xml", type=Path, default=None, help="scene path (default: go2_flat_scene.xml)")
    parser.add_argument("--lunar", action="store_true",
                        help="use Go2LunarEnv (terrain-relative height) on the lunar hfield scene "
                             "(default xml go2_lunar_scene.xml). Phase 2.")
    parser.add_argument("--total-timesteps", type=int, default=3_500_000,
                        help="default 3.5M ~= converged (>=97%% of the 5M reward; 5M adds only ~2-3pp). "
                             "Use 2.5M for fast config screening (~93%%). See RESULTS.md convergence note.")
    parser.add_argument("--num-envs", type=int, default=4) # DummyVecEnv 串行无益；--subproc 下设 ~核数-2
    parser.add_argument("--subproc", action="store_true",
                        help="用 SubprocVecEnv：每个环境一个子进程 → 物理跨多核并行（吃满 CPU）。"
                             "月面 hfield 单核慢，强烈建议配 --num-envs 12 用满 14 核机。"
                             "默认 DummyVecEnv（单进程串行，只用 1-2 核，向后兼容）。")
    parser.add_argument("--torch-threads", type=int, default=0,
                        help="主进程 PPO 更新的 torch 线程数（0=不设，用 torch 默认）。"
                             "配 --subproc 时建议 4-8，避免与子进程抢核。")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="go2_walk")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=250_000,
                        help="save a checkpoint every N total env steps (across all envs)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="warm-start: load policy weights from this .zip and keep training "
                             "(num_timesteps still resets, so it trains a full --total-timesteps).")
    parser.add_argument("--init-vecnorm", type=str, default=None,
                        help="warm-start: seed VecNormalize obs/reward stats from this .pkl. "
                             "Use together with --init-from so the loaded policy sees matching "
                             "normalization from step 0 (otherwise the warm policy gets garbage obs).")
    parser.add_argument("--no-progress-bar", action="store_true",
                        help="disable the tqdm progress bar (cleaner logs for detached background runs; "
                             "verbose=1 still prints the rollout table each update).")
    args = parser.parse_args()

    if args.torch_threads > 0:
        import torch
        torch.set_num_threads(args.torch_threads)

    run_dir = Path("runs") / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(args.xml, args.seed, i, args.lunar) for i in range(args.num_envs)]
    if args.subproc and args.num_envs > 1:
        # 每个环境一个子进程 → MuJoCo 物理跨核并行。Windows 用 spawn；闭包由 SB3 的
        # cloudpickle 包装传入子进程（make_env 只捕获 Path/int/bool，可序列化）。
        env = SubprocVecEnv(env_fns)  # start_method 默认 spawn (Windows)
        print(f"SubprocVecEnv: {args.num_envs} 子进程并行")
    else:
        env = DummyVecEnv(env_fns)
    if args.init_vecnorm:
        # Continue from a previous run's running statistics (e.g. Phase-1 N) so the
        # warm-started policy is fed obs on the scale it was trained on. training=True
        # lets the stats keep adapting to the new (payload) dynamics.
        env = VecNormalize.load(args.init_vecnorm, env)
        env.training = True
        env.norm_reward = True
        print(f"Seeded VecNormalize stats from {args.init_vecnorm}")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if args.init_from:
        # Transfer learning: reuse the converged policy/value nets and PPO hyperparameters
        # from a prior run, then keep optimizing on the new env. Only the env changed
        # (same obs=54 / act=12 spaces), so the networks load directly.
        print(f"Warm-starting policy from {args.init_from}")
        model = PPO.load(
            args.init_from,
            env=env,
            device=args.device,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=1.0,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
        )

    # CheckpointCallback counts vec-env steps (n_calls), so divide the desired
    # total-step cadence by num_envs. With defaults: 250k/4 -> save every 62.5k
    # calls = every 250k total steps (ckpts at 0.25M, 0.5M, ..., 5M).
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_go2",
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=not args.no_progress_bar,
        reset_num_timesteps=True,
    )

    model.save(run_dir / "ppo_go2_final")
    env.save(run_dir / "vecnormalize.pkl")
    env.close()
    print(f"Saved model and normalization stats to {run_dir}")


if __name__ == "__main__":
    main()
