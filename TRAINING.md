# Go2 MuJoCo Locomotion Training (Phase 1 ✅ done → Phase 3: add Z1 arm, then Phase 2 lunar)

PPO training setup for the Unitree Go2 on flat ground, ported from
`TRAINING-Aliengo/`. Same Stable-Baselines3 + MuJoCo stack; the model is the
native Go2 MJCF (`go2.xml`) with its torque motors swapped for position-servo
(PD) actuators, so `ctrl=0` holds the `stand` keyframe and the policy action is
an offset around that pose.

## Files

- `go2_env.py` — Gymnasium env (`Go2WalkEnv`): 54-d obs, 12-d action, walking reward.
- `train_ppo.py` — PPO training entry point (Stable-Baselines3).
- `play_policy.py` — load and replay a trained policy in the MuJoCo viewer.
- `eval_policy.py` — headless QUANTITATIVE acceptance: fixed-command rollouts,
  metrics + thresholds + gait plot / contact sheet / gif.
- `smoke_test.py` — random-action check: env loads, steps, 54-d finite obs/reward.
- `diagnostics/smoke_stand.py` — standing check: `ctrl=0` holds the stance (PD gain sanity).
- `diagnostics/inspect_model.py` — dumps `go2.xml` structure (actuator/joint/keyframe/foot order).
- `go2.xml`, `go2_flat_scene.xml`, `go2_assets/` — model + flat scene + meshes.
- `requirements.txt` — Python dependencies.

## Install

```powershell
python -m pip install -r requirements.txt
```

CPU is fine (~1450 fps here). For a CUDA build, install your preferred PyTorch
first, then run the requirements command.

## Quick checks

```powershell
python diagnostics/inspect_model.py   # model loads; prints actuator/joint/keyframe facts
python diagnostics/smoke_stand.py     # ctrl=0 holds the stand pose (validate kp/kd)
python smoke_test.py                  # 200 random steps; 54-d finite obs/reward
```

> **Windows + non-ASCII path:** this repo lives under a Chinese path and MuJoCo's
> C++ parser cannot open such absolute paths. Every script `os.chdir()`s to its
> own folder and loads a relative ASCII filename, so **run the scripts from inside
> `TRAINING-Go2/`** (the meshes resolve relative to the cwd too).

## Train

```powershell
# health check first
python train_ppo.py --total-timesteps 100000 --num-envs 4

# full run (~22 min on CPU; 3.5M is the default)
python train_ppo.py --total-timesteps 3500000 --num-envs 4 `
    --run-name my_run --checkpoint-freq 250000
```

Outputs → `runs/<run-name>/`:

- `ppo_go2_final.zip` + `vecnormalize.pkl` — keep both; replay and eval need the
  normalization statistics.
- `checkpoints/ppo_go2_<N>_steps.zip` (+ matching `vecnormalize`) every
  `--checkpoint-freq` total steps.
- `tensorboard/` — view with `tensorboard --logdir runs/<run-name>/tensorboard`.

`--checkpoint-freq` is in **total** env steps; the script divides by `--num-envs`
for SB3's per-vec-step counter (e.g. `250000` with 4 envs → a checkpoint every
0.25 M steps: 0.25 M, 0.5 M, …, 5 M).

## Replay

```powershell
python play_policy.py --model runs/go2_gN_tc55_3p5M/ppo_go2_final.zip `
    --vecnormalize runs/go2_gN_tc55_3p5M/vecnormalize.pkl
```

## Evaluate (quantitative acceptance)

```powershell
python eval_policy.py --run go2_gN_tc55_3p5M            # metrics + gait plot
python eval_policy.py --run go2_gN_tc55_3p5M --render   # + contact sheet + gif
```

Runs 5 deterministic fixed-command episodes (fwd 0.3 / 0.5 / 0.8, fwd+yaw,
lateral), scores velocity-tracking MAE, per-foot duty factor, flight fraction,
action/torque saturation, roll/pitch std and base height against `THRESHOLDS`,
and writes `eval_out/<run>/metrics.json` (+ `contact_schedule.png`, and with
`--render` `contact_sheet.png` / `rollout.gif`). It first prints a spawn
diagnostic (foot gaps at the `stand` keyframe) to catch a wrong spawn height.

## Notes

- **PD actuators:** `go2.xml` uses `gaintype=fixed biastype=affine`, so
  `force = kp·(ctrl + q_stand − q) − kd·q̇` and `ctrl=0` holds the stand pose.
  The action is an offset around stand, clipped to each actuator's `ctrlrange`.
- **Final config (Phase 1, v1.4):** kp=35/kd=0.75, stand base_z=0.38, legs
  `[0,0.8,-1.5]`; `action_scale` selects the three shipped configs — **N**
  `[0.125,0.55,0.55]` (live default), **J/G** `[0.125,0.45,0.45]` (5M / 3.5M).
  All walk a symmetric trot, never fall, action_sat 0.66–0.72. See `RESULTS.md`
  for the full Gen A–O history.
- **Curriculum idea:** start forward-only with a narrow command range, then add
  yaw and lateral commands; tighten posture / add terrain last.
- **Keep `vecnormalize.pkl` with its model** — replay/eval are wrong without the
  matching normalization statistics.
```

