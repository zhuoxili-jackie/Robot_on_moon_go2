"""Headless QUANTITATIVE evaluation of a trained Go2 PPO policy.

Adapted from ref/eval_policy_old.py (which targeted the aliengo+z1 lunar env).
Runs deterministic fixed-command episodes and turns acceptance from "looks OK by
eye" into numeric thresholds:

  * velocity tracking error (vx / vy / yaw vs command)
  * gait duty factor per foot + touchdowns/sec
  * action & torque saturation fraction
  * flight fraction (all four feet airborne) + all-four-stance fraction
  * posture (roll/pitch std, base height)
and saves a gait contact-schedule plot (+ optional rendered contact sheet / gif).

It also prints a SPAWN DIAGNOSTIC: foot gaps at the 'stand' keyframe, guarding the
"wrong spawn height -> robot keeps falling" bug (see CLAUDE.md / lunar hfield note).

Usage:
    python eval_policy.py --run go2_walk            # evaluate runs/go2_walk
    python eval_policy.py --run go2_walk --render   # also dump rendered frames
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from go2_env import Go2WalkEnv
from play_policy import patch_sb3_zip_loader


# Acceptance thresholds (INITIAL values -- calibrate after a real training run).
THRESHOLDS = {
    "vx_track_mae_max": 0.12,        # mean |vx - cmd_vx|, m/s
    "vyaw_track_mae_max": 0.25,      # mean |wz - cmd_yaw|, rad/s
    "flight_frac_max": 0.35,         # fraction of time all 4 feet airborne
    "force_sat_frac_max": 0.10,      # fraction of (step,joint) at >=97% torque limit
    "action_sat_frac_max": 0.50,     # fraction of |action| > 0.95  (calibrated from go2_best ~0.4)
    "pitch_std_deg_max": 8.0,
    "roll_std_deg_max": 8.0,
    "duty_min": 0.25,                # per-foot stance fraction (flying trot can dip below 0.5)
    "duty_max": 0.85,
}


def euler_from_quat(quat: np.ndarray) -> tuple[float, float]:
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    pitch = -math.asin(np.clip(r[2, 0], -1.0, 1.0))
    roll = math.atan2(r[2, 1], r[2, 2])
    return roll, pitch


def ground_height(env: Go2WalkEnv) -> float:
    """Surface z under the robot. Flat scene: the 'lunar_terrain' plane's z (0).
    Phase 2 (hfield): replace with a height-field lookup at (x, y)."""
    gid = env.terrain_geom_id
    if gid < 0:
        return 0.0
    return float(env.model.geom_pos[gid, 2])


def spawn_diagnostic(env: Go2WalkEnv) -> dict:
    """Reset to 'stand' and measure foot gaps to the ground. Feet should sit a few
    mm above the surface (small positive gap, no deep penetration). A large gap =>
    robot will drop at reset; deep penetration => physics blow-up."""
    mujoco.mj_resetDataKeyframe(env.model, env.data, env.stand_key_id)
    mujoco.mj_forward(env.model, env.data)
    foot_z = env.data.geom_xpos[env.foot_geom_ids, 2]
    gaps = foot_z - ground_height(env)
    return {
        "stand_base_z": round(float(env.data.qpos[2]), 3),
        "ground_z": round(ground_height(env), 3),
        "foot_gaps_m": [round(float(g), 3) for g in gaps],
        "min_foot_gap_m": round(float(gaps.min()), 3),
        "max_penetration_m": round(float(min(gaps.min(), 0.0)), 3),
        "ok": bool(gaps.min() > -0.02 and gaps.min() < 0.10),
    }


def run_episode(env, vec_norm, model, command, seed, render_dir=None, render_every=4):
    obs, _ = env.reset(seed=seed, options={"command": command})
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    renderer = cam = None
    frames: list[np.ndarray] = []
    if render_dir is not None:
        try:
            renderer = mujoco.Renderer(env.model, height=360, width=480)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = base_id
            cam.distance, cam.azimuth, cam.elevation = 2.0, 145, -12
        except Exception as exc:  # GL context unavailable
            print(f"  [render disabled: {type(exc).__name__}: {exc}]")
            renderer = None

    log = {k: [] for k in ("vx", "vy", "wz", "height", "contacts", "action",
                           "joint_pos", "roll", "pitch", "force_sat")}
    start_xy = env.data.qpos[0:2].copy()
    terminated = truncated = False
    steps = 0
    for step in range(env.max_steps):
        norm_obs = vec_norm.normalize_obs(obs[None, :].astype(np.float32))
        action, _ = model.predict(norm_obs, deterministic=True)
        action = action[0]
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1

        lin, ang = env._base_velocity_body()
        roll, pitch = euler_from_quat(env.data.qpos[3:7])
        force = np.abs(env.data.actuator_force[env.actuator_ids])
        force_sat = float(np.mean(force >= 0.97 * env.model.actuator_forcerange[env.actuator_ids, 1]))
        log["vx"].append(lin[0]); log["vy"].append(lin[1]); log["wz"].append(ang[2])
        log["height"].append(float(env.data.qpos[2]))
        log["contacts"].append(env._foot_contacts().copy())
        log["action"].append(action.copy())
        log["joint_pos"].append(env.data.qpos[env.joint_qpos_adr].copy())
        log["roll"].append(roll); log["pitch"].append(pitch)
        log["force_sat"].append(force_sat)

        if renderer is not None and step % render_every == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
        if terminated or truncated:
            break
    if renderer is not None:
        renderer.close()

    contacts = np.array(log["contacts"])      # (T, 4)
    actions = np.array(log["action"])         # (T, 12)
    duration = steps * env.dt
    rising = np.maximum(np.diff(contacts, axis=0), 0).sum(axis=0)  # touchdowns/foot
    duty = contacts.mean(axis=0)
    cmd = command.astype(float)

    metrics = {
        "command": cmd.tolist(),
        "steps": steps,
        "seconds": round(duration, 2),
        "terminated": bool(terminated),
        "no_fall": bool(not terminated),
        "mean_vx": round(float(np.mean(log["vx"])), 3),
        "mean_vy": round(float(np.mean(log["vy"])), 3),
        "mean_wz": round(float(np.mean(log["wz"])), 3),
        "vx_track_mae": round(float(np.mean(np.abs(np.array(log["vx"]) - cmd[0]))), 3),
        "vy_track_mae": round(float(np.mean(np.abs(np.array(log["vy"]) - cmd[1]))), 3),
        "vyaw_track_mae": round(float(np.mean(np.abs(np.array(log["wz"]) - cmd[2]))), 3),
        "xy_distance": round(float(np.linalg.norm(env.data.qpos[0:2] - start_xy)), 2),
        "mean_height": round(float(np.mean(log["height"])), 3),
        "duty_factor_per_foot": [round(float(d), 2) for d in duty],
        "touchdowns_per_sec": [round(float(r / duration), 2) for r in rising],
        "all4_stance_frac": round(float((contacts.sum(axis=1) == 4).mean()), 2),
        "flight_frac": round(float((contacts.sum(axis=1) == 0).mean()), 2),
        "action_sat_frac": round(float((np.abs(actions) > 0.95).mean()), 3),
        "mean_abs_action": round(float(np.abs(actions).mean()), 3),
        "pitch_std_deg": round(float(np.degrees(np.std(log["pitch"]))), 1),
        "roll_std_deg": round(float(np.degrees(np.std(log["roll"]))), 1),
        "force_sat_frac": round(float(np.mean(log["force_sat"])), 3),
    }
    metrics["checks"], metrics["pass"] = evaluate_thresholds(metrics)

    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)
        save_contact_schedule(contacts, env.dt, render_dir / "contact_schedule.png")
        if frames:
            save_frames(frames, render_dir, fps=1.0 / (env.dt * render_every))
    return metrics


def evaluate_thresholds(m: dict) -> tuple[dict, bool]:
    t = THRESHOLDS
    checks = {
        "no_fall": m["no_fall"],
        "vx_track": m["vx_track_mae"] <= t["vx_track_mae_max"],
        "vyaw_track": m["vyaw_track_mae"] <= t["vyaw_track_mae_max"],
        "flight": m["flight_frac"] <= t["flight_frac_max"],
        "force_sat": m["force_sat_frac"] <= t["force_sat_frac_max"],
        "action_sat": m["action_sat_frac"] <= t["action_sat_frac_max"],
        "pitch": m["pitch_std_deg"] <= t["pitch_std_deg_max"],
        "roll": m["roll_std_deg"] <= t["roll_std_deg_max"],
        "duty": all(t["duty_min"] <= d <= t["duty_max"] for d in m["duty_factor_per_foot"]),
    }
    return checks, bool(all(checks.values()))


def save_contact_schedule(contacts: np.ndarray, dt: float, path: Path) -> None:
    """Gait diagram: foot-contact raster (4 feet x time). Headless-safe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = contacts.shape[0]
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.imshow(contacts.T, aspect="auto", cmap="Greys", interpolation="nearest",
              extent=[0, T * dt, 3.5, -0.5])
    ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["FR", "FL", "RR", "RL"])
    ax.set_xlabel("time (s)"); ax.set_title("foot contact schedule (black = stance)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_frames(frames: list[np.ndarray], out_dir: Path, fps: float) -> None:
    from PIL import Image

    sheet_frames = frames[::5][:30]
    cols = 5
    rows = math.ceil(len(sheet_frames) / cols)
    h, w, _ = sheet_frames[0].shape
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, fr in enumerate(sheet_frames):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = fr
    Image.fromarray(sheet).save(out_dir / "contact_sheet.png")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out_dir / "rollout.gif", save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=None, help="scene path (default: go2_flat_scene.xml)")
    parser.add_argument("--run", type=str, default="go2_walk")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--render", action="store_true", help="also dump rendered frames")
    parser.add_argument("--model", type=str, default=None,
                        help="explicit policy .zip to evaluate (default: runs/<run>/ppo_go2_final.zip)")
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="explicit VecNormalize .pkl (default: runs/<run>/vecnormalize.pkl)")
    args = parser.parse_args()

    run_dir = Path("runs") / args.run
    out_dir = args.out or Path("eval_out") / args.run

    model_path = args.model or str(run_dir / "ppo_go2_final.zip")
    vecnorm_path = args.vecnormalize or str(run_dir / "vecnormalize.pkl")
    print(f"== loading policy: {model_path}")
    print(f"== loading vecnormalize: {vecnorm_path}")

    patch_sb3_zip_loader()
    env = Go2WalkEnv(xml_path=args.xml)
    vec = DummyVecEnv([lambda: env])
    vec_norm = VecNormalize.load(vecnorm_path, vec)
    vec_norm.training = False
    vec_norm.norm_reward = False
    model = PPO.load(model_path, device="cpu")

    print("== model / scene sanity ==")
    print("gravity:", env.model.opt.gravity, " total mass:", round(float(env.model.body_mass.sum()), 2), "kg")
    spawn = spawn_diagnostic(env)
    print("spawn @ 'stand':", spawn)
    if not spawn["ok"]:
        print("  !! spawn height looks wrong (feet floating or penetrating) -- fix before trusting training")

    episodes = [
        ("fwd_0.3", np.array([0.3, 0.0, 0.0], dtype=np.float32), 10),
        ("fwd_0.5", np.array([0.5, 0.0, 0.0], dtype=np.float32), 11),
        ("fwd_0.8", np.array([0.8, 0.0, 0.0], dtype=np.float32), 12),
        ("fwd_yaw", np.array([0.5, 0.0, 0.4], dtype=np.float32), 13),
        ("lateral", np.array([0.4, -0.2, 0.0], dtype=np.float32), 14),
    ]
    results = {"spawn": spawn}
    n_pass = 0
    for name, cmd, seed in episodes:
        rdir = out_dir / name
        metrics = run_episode(env, vec_norm, model, cmd, seed, rdir)
        results[name] = metrics
        n_pass += int(metrics["pass"])
        print(f"\n== {name} cmd={cmd.tolist()}  PASS={metrics['pass']} ==")
        for k in ("no_fall", "vx_track_mae", "vyaw_track_mae", "duty_factor_per_foot",
                  "touchdowns_per_sec", "flight_frac", "force_sat_frac", "action_sat_frac",
                  "pitch_std_deg", "roll_std_deg", "mean_height"):
            print(f"  {k}: {metrics[k]}")
        print(f"  failed checks: {[k for k, v in metrics['checks'].items() if not v]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\n=== OVERALL: {n_pass}/{len(episodes)} episodes pass thresholds ===")
    print(f"Saved metrics + gait plots to {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
