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
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=4) # 这里可以增加至64、1024等
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="go2_walk")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=250_000,
                        help="save a checkpoint every N total env steps (across all envs)")
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(args.xml, args.seed, i) for i in range(args.num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

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
        progress_bar=True,
    )

    model.save(run_dir / "ppo_go2_final")
    env.save(run_dir / "vecnormalize.pkl")
    env.close()
    print(f"Saved model and normalization stats to {run_dir}")


if __name__ == "__main__":
    main()
