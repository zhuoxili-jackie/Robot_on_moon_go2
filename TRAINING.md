# Go2 MuJoCo Locomotion Training (Phase 1 ✅ → Phase 3 ✅ → Phase 2 ✅ lunar → Phase 4 keyboard)

PPO training setup for the Unitree Go2 on flat ground, ported from
`TRAINING-Aliengo/`. Same Stable-Baselines3 + MuJoCo stack; the model is the
native Go2 MJCF (`go2.xml`) with its torque motors swapped for position-servo
(PD) actuators, so `ctrl=0` holds the `stand` keyframe and the policy action is
an offset around that pose.

## Files

- `go2_env.py` — Gymnasium env (`Go2WalkEnv`): 54-d obs, 12-d action, walking reward.
- `go2_lunar_env.py` — `Go2LunarEnv(Go2WalkEnv)` (Phase 2): terrain-relative height via
  downward `mj_ray` on the hfield + cross-terrain spawn. Same 54-d obs (blind, no height scan).
- `train_ppo.py` — PPO training entry point (Stable-Baselines3). Flags: `--lunar`,
  `--subproc` (multi-core SubprocVecEnv), `--torch-threads`, `--init-from`/`--init-vecnorm` (warm-start).
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

# full flat run (~22 min on CPU; 3.5M is the default)
python train_ppo.py --total-timesteps 3500000 --num-envs 4 `
    --run-name my_run --checkpoint-freq 250000

# ★ MULTI-CORE (strongly recommended): SubprocVecEnv runs one OS process per env so
#   MuJoCo physics parallelizes across cores. DummyVecEnv (default) is serial → only
#   1-2 cores. On a 14-core box, --num-envs 12 uses ~9 cores, ~4.7x faster (lunar 3.5M
#   ~100 min → ~20 min). --torch-threads keeps the PPO update from fighting the env procs.
python train_ppo.py --subproc --num-envs 12 --torch-threads 4 `
    --total-timesteps 3500000 --run-name my_run --checkpoint-freq 250000

# ★ LUNAR (Phase 2): hfield terrain + terrain-relative env, warm-started from the flat run.
python train_ppo.py --lunar --subproc --num-envs 12 --torch-threads 4 `
    --total-timesteps 3500000 --run-name my_lunar --checkpoint-freq 250000 `
    --init-from runs/go2_z1_flat_final_3p5M/ppo_go2_final.zip `
    --init-vecnorm runs/go2_z1_flat_final_3p5M/vecnormalize.pkl
```

> GPU/MJX is **not** an option for the lunar scene: MJX doesn't support hfield collision.
> CPU `--subproc` is the right way to use the cores. Regenerate the terrain with
> `python lunar_assets/make_lunar_hfield.py` (then sync `go2_lunar_scene.xml`'s hfield
> `size`/`pos` to the new `lunar_assets/lunar_designed_meta.json`).

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
python eval_policy.py --lunar --run go2_lunar_3p5M --render   # ★ lunar: relaxed thresholds + crater/hill traversal episodes
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

