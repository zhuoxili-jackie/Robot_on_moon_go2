"""Short random-action check for the Go2 walking environment.

Phase 1 / Step 2 acceptance: confirm Go2WalkEnv loads, resets, steps, and returns
a 52-d observation with finite rewards. Run:

    python smoke_test.py
"""

from __future__ import annotations

import argparse

import numpy as np

from go2_env import Go2WalkEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short random-action environment check.")
    parser.add_argument("--xml", default=None, help="scene path (default: go2_flat_scene.xml next to env)")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    env = Go2WalkEnv(xml_path=args.xml)
    print(f"obs_space={env.observation_space.shape}  act_space={env.action_space.shape}")
    print(f"dt={env.dt:.3f}s  max_steps={env.max_steps}  stand_height={env.stand_height:.3f}")
    print(f"stand_joint_qpos (actuator order)={np.round(env.stand_joint_qpos, 3).tolist()}")

    obs, info = env.reset(seed=0, options={"command": np.array([0.4, 0.0, 0.0], dtype=np.float32)})
    print(f"reset obs_shape={obs.shape}  finite={np.isfinite(obs).all()}")

    total_reward = 0.0
    rewards = []
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        if terminated or truncated:
            print(f"episode ended at step={step} terminated={terminated} truncated={truncated}")
            break

    rewards = np.asarray(rewards)
    print(f"final obs_shape={obs.shape}  obs_finite={np.isfinite(obs).all()}")
    print(f"total_reward={total_reward:.3f}  per_step[min/mean/max]="
          f"{rewards.min():.3f}/{rewards.mean():.3f}/{rewards.max():.3f}")
    print(f"last_info={ {k: round(v, 3) for k, v in info.items()} }")

    ok = obs.shape == (52,) and np.isfinite(rewards).all()
    print("RESULT:", "PASS - env loads, steps, 52-d finite obs/reward." if ok else "FAIL")
    env.close()


if __name__ == "__main__":
    main()
