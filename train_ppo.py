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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from go2_env import Go2WalkEnv


def make_env(xml_path: Path | None, seed: int, rank: int):
    def _init():
        env = Go2WalkEnv(xml_path=xml_path)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Go2 walking with PPO.")
    parser.add_argument("--xml", type=Path, default=None, help="scene path (default: go2_flat_scene.xml)")
    parser.add_argument("--total-timesteps", type=int, default=3_500_000,
                        help="default 3.5M ~= converged (>=97%% of the 5M reward; 5M adds only ~2-3pp). "
                             "Use 2.5M for fast config screening (~93%%). See RESULTS.md convergence note.")
    parser.add_argument("--num-envs", type=int, default=4) # 这里可以增加至64、1024等
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

    run_dir = Path("runs") / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(args.xml, args.seed, i) for i in range(args.num_envs)])
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
