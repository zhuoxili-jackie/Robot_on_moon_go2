"""Gen N go2_env — thigh/calf scale 同步小幅扩大：0.45 → 0.55.

Gen H/I: [0.125,0.65,0.65] → 双腿退化步态（策略找到"一对角始终踩地"捷径）
Gen G/J: [0.125,0.45,0.45] → 步态完美但 action_sat=0.67-0.73（饱和临界 0.474，0.45<0.474）
Gen M:   [0.125,0.45,0.65] → lateral 崩溃（calf 顶高机身 0.387>stand 0.38→重心过高→侧移不稳）

Gen N 假设：0.55 = 既超过饱和临界（0.55>0.474），又远低于双腿退化阈值（0.55«0.65）的甜蜜点。
  - thigh/calf 偏移 0.43 rad（典型动态步态值）→ action=0.43/0.55=0.78（不顶满 <0.95）
  - 预期 action_sat 从 0.67 降至 0.35-0.50（饱和事件从必发变为偶发/不发）
  - 与 Gen H/I 的对称扩大策略相同，但幅度更保守（0.55 vs 0.65）

go2.xml: kp=35/kd=0.75（Gen G/J 状态，不变）。
action_size: -0.005（原始值，Gen L 证明 -0.02 破坏侧移稳定性，不再尝试强惩罚）。
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


LEG_ACTUATORS = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)

FOOT_GEOMS = ("FR", "FL", "RR", "RL")
DEFAULT_SCENE = Path(__file__).with_name("go2_flat_scene.xml")


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _load_model(xml_path: Path) -> mujoco.MjModel:
    xml = Path(xml_path)
    model_dir = xml.resolve().parent
    prev = os.getcwd()
    try:
        os.chdir(model_dir)
        return mujoco.MjModel.from_xml_path(xml.name)
    finally:
        os.chdir(prev)


class Go2WalkEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        frame_skip: int = 10,
        episode_seconds: float = 12.0,
        command_range=((0.15, 0.70), (-0.10, 0.10), (-0.30, 0.30)),
        gait_cycle_seconds: float = 0.5,
        gait_clock_scale: float = 1.0,
        trot_reward_weight: float = 0.35,
        same_side_contact_penalty_weight: float = 0.25,
        render_mode: str | None = None,
        min_base_height: float = 0.16,
    ) -> None:
        self.xml_path = Path(xml_path) if xml_path is not None else DEFAULT_SCENE
        self.model = _load_model(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = int(episode_seconds / self.dt)
        self.command_range = command_range
        self.gait_cycle_seconds = float(gait_cycle_seconds)
        self.gait_clock_scale = float(gait_clock_scale)
        self.trot_reward_weight = float(trot_reward_weight)
        self.same_side_contact_penalty_weight = float(same_side_contact_penalty_weight)
        self.render_mode = render_mode

        self.actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in LEG_ACTUATORS],
            dtype=np.int32,
        )
        joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        self.joint_qpos_adr = self.model.jnt_qposadr[joint_ids].astype(np.int32)
        self.joint_dof_adr = self.model.jnt_dofadr[joint_ids].astype(np.int32)

        self.foot_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOMS],
            dtype=np.int32,
        )
        self.terrain_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "lunar_terrain"
        )

        self.action_scale = np.array([0.125, 0.55, 0.55] * 4, dtype=np.float32)
        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].astype(np.float32)

        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if self.stand_key_id < 0:
            raise ValueError("Expected a keyframe named 'stand'")
        self.stand_qpos = self.model.key_qpos[self.stand_key_id].copy()
        self.stand_joint_qpos = self.stand_qpos[self.joint_qpos_adr].astype(np.float32)
        self.stand_height = float(self.stand_qpos[2])
        self.min_base_height = float(min_base_height)

        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(54,), dtype=np.float32)
        self.viewer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)
        joint_noise = self.np_random.uniform(-0.03, 0.03, size=12)
        self.data.qpos[self.joint_qpos_adr] = self.stand_joint_qpos + joint_noise
        self.data.qvel[:] = self.np_random.uniform(-0.02, 0.02, size=self.model.nv)
        self.data.ctrl[self.actuator_ids] = 0.0
        if options and "command" in options:
            self.command = np.asarray(options["command"], dtype=np.float32)
        else:
            self.command = self._sample_command()
        self.last_action.fill(0.0)
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        ctrl = np.clip(action * self.action_scale, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[self.actuator_ids] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward, reward_terms = self._reward(action)
        self.step_count += 1
        terminated = self._is_unhealthy()
        truncated = self.step_count >= self.max_steps
        self.last_action = action.copy()
        info = self._get_info()
        info.update(reward_terms)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _sample_command(self):
        ranges = np.asarray(self.command_range, dtype=np.float32)
        return self.np_random.uniform(ranges[:, 0], ranges[:, 1]).astype(np.float32)

    def _base_rotation(self):
        return _quat_to_matrix(self.data.qpos[3:7])

    def _projected_gravity(self):
        return self._base_rotation().T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def _base_velocity_body(self):
        rotation = self._base_rotation()
        return rotation.T @ self.data.qvel[0:3], rotation.T @ self.data.qvel[3:6]

    def _foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(4, dtype=np.float32)
        if self.terrain_geom_id < 0:
            return contacts
        foot_to_index = {int(gid): idx for idx, gid in enumerate(self.foot_geom_ids)}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self.terrain_geom_id and g2 in foot_to_index:
                contacts[foot_to_index[g2]] = 1.0
            elif g2 == self.terrain_geom_id and g1 in foot_to_index:
                contacts[foot_to_index[g1]] = 1.0
        return contacts

    def _nonfoot_ground_contact_penalty(self) -> float:
        """Gen E: teacher's implementation — unique nonfoot geom IDs, uncapped, weight 2.5."""
        if self.terrain_geom_id < 0:
            return 0.0
        foot_set = {int(g) for g in self.foot_geom_ids}
        nonfoot_geoms: set[int] = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self.terrain_geom_id and g2 not in foot_set:
                nonfoot_geoms.add(g2)
            elif g2 == self.terrain_geom_id and g1 not in foot_set:
                nonfoot_geoms.add(g1)
        return float(len(nonfoot_geoms))

    def _gait_phase(self) -> float:
        return float((self.step_count * self.dt / self.gait_cycle_seconds) % 1.0)

    def _gait_clock(self) -> np.ndarray:
        angle = 2.0 * np.pi * self._gait_phase()
        return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32) * self.gait_clock_scale

    def _desired_trot_contacts(self) -> np.ndarray:
        if self._gait_phase() < 0.5:
            return np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        projected_gravity = self._projected_gravity()
        base_linear, base_angular = self._base_velocity_body()
        joint_pos = self.data.qpos[self.joint_qpos_adr] - self.stand_joint_qpos
        joint_vel = self.data.qvel[self.joint_dof_adr]
        contacts = self._foot_contacts()
        gait_clock = self._gait_clock()
        obs = np.concatenate([
            projected_gravity, base_linear, base_angular,
            self.command, gait_clock,
            joint_pos, joint_vel, self.last_action, contacts,
        ])
        return obs.astype(np.float32)

    def _reward(self, action: np.ndarray):
        base_linear, base_angular = self._base_velocity_body()
        vel_err = np.array([
            base_linear[0] - self.command[0],
            base_linear[1] - self.command[1],
            base_angular[2] - self.command[2],
        ], dtype=np.float64)
        tracking = float(np.exp(-np.dot(vel_err, vel_err) / 0.25))

        projected_gravity = self._projected_gravity()
        upright = float(np.clip(-projected_gravity[2], 0.0, 1.0))
        height_err = abs(float(self.data.qpos[2]) - self.stand_height)
        height = float(np.exp(-(height_err * height_err) / 0.025))  # Gen E: sigma^2 0.030 -> 0.025

        contacts = self._foot_contacts()
        desired = self._desired_trot_contacts()
        trot_match = float(np.exp(-np.sum(np.square(contacts - desired)) / 0.25))
        same_side = float(contacts[0] * contacts[1] + contacts[2] * contacts[3])
        all_off = 1.0 if float(np.sum(contacts)) == 0.0 else 0.0  # Gen E: all-4-off flight penalty

        action_rate = float(np.sum(np.square(action - self.last_action)))
        action_size = float(np.sum(np.square(action)))
        joint_speed = float(np.sum(np.square(self.data.qvel[self.joint_dof_adr])))
        nonfoot = self._nonfoot_ground_contact_penalty()
        unhealthy = 1.0 if self._is_unhealthy() else 0.0

        reward = (
            2.0 * tracking
            + 0.60 * upright
            + 0.45 * height
            + self.trot_reward_weight * trot_match
            - self.same_side_contact_penalty_weight * same_side
            - 0.5 * all_off                # Gen E: explicit anti-pronk
            - 0.03 * action_rate
            - 0.005 * action_size
            - 0.0005 * joint_speed
            - 2.5 * nonfoot                # Gen E: 2.0 -> 2.5, unique-geom uncapped
            - 2.0 * unhealthy
        )

        return float(reward), {
            "reward_tracking": tracking,
            "reward_upright": upright,
            "reward_height": height,
            "reward_trot_match": trot_match,
            "penalty_same_side_contact": same_side,
            "penalty_all_off": all_off,
            "penalty_action_rate": action_rate,
            "penalty_action_size": action_size,
            "penalty_joint_speed": joint_speed,
            "penalty_nonfoot_contact": nonfoot,
        }

    def _is_unhealthy(self) -> bool:
        projected_gravity = self._projected_gravity()
        too_low = float(self.data.qpos[2]) < self.min_base_height
        tipped = projected_gravity[2] > -0.35
        bad_number = not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
        return bool(too_low or tipped or bad_number)

    def _get_info(self) -> dict[str, float]:
        base_linear, base_angular = self._base_velocity_body()
        return {
            "command_vx": float(self.command[0]),
            "command_vy": float(self.command[1]),
            "command_yaw": float(self.command[2]),
            "gait_phase": float(self._gait_phase()),
            "base_vx": float(base_linear[0]),
            "base_vy": float(base_linear[1]),
            "base_yaw_rate": float(base_angular[2]),
            "base_height": float(self.data.qpos[2]),
        }
