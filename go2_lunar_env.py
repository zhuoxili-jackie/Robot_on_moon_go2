"""Go2LunarEnv —— Phase 2 月面环境（薄子类，复用 Go2WalkEnv 的全部 RL 逻辑）。

与平地 Go2WalkEnv 的唯一区别：高度全部改成「相对当地地表」，否则狗走进大坑里
base 的绝对世界 z 掉破阈值会被**误判摔倒**、height 奖励也会错（见 CONTINUATION §3.3 的 L3）。

做法（最小侵入）：
  * 覆写 `_ground_height()`：向下投射 mj_ray 命中地形（hfield）得到 base 正下方地表 z。
    用 geomgroup 只收 group-0 的 'lunar_terrain'（机器人碰撞/视觉是 group 3/2 → 被过滤），
    所以射线只打地形、不打狗自身，**与 hfield 朝向无关**（用 MuJoCo 自己的碰撞面，最稳）。
  * 覆写 `reset()`：把出生点随机散布到地形上（默认半径 spawn_radius 内、坡度 ≤ max_spawn_slope
    的平缓处，拒绝采样），base_z = 当地地表 z + 站高 → 出生即贴地（复刻平台出生的足端间隙）。
    传 options={"spawn": (x,y)} 可指定出生点（eval 用来定点测某个坑/坡）。

ray 地表查找也导出为模块函数 `ray_terrain_height`，供 eval_policy 复用。
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from go2_env import Go2WalkEnv

LUNAR_SCENE = Path(__file__).with_name("go2_lunar_scene.xml")

# 射线只收 group 0（地形）；机器人碰撞 group3 / 视觉 group2 被过滤掉。
_TERRAIN_GROUP_MASK = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
_RAY_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_RAY_START_Z = 8.0  # 远高于任何地表（设计地形 max≈+0.33）


def ray_terrain_height(model, data, x: float, y: float,
                       terrain_gid: int = -1) -> float | None:
    """从 (x, y, 8) 向下投射，返回地表世界 z；未命中返回 None。"""
    pnt = np.array([x, y, _RAY_START_Z], dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, _RAY_DOWN, _TERRAIN_GROUP_MASK,
                         1, -1, geomid)  # flg_static=1（地形是静态体）
    if dist < 0:
        return None
    if terrain_gid >= 0 and int(geomid[0]) != terrain_gid:
        return None
    return _RAY_START_Z - float(dist)


class Go2LunarEnv(Go2WalkEnv):
    def __init__(
        self,
        xml_path=None,
        spawn_radius: float = 4.5,
        max_spawn_slope_deg: float = 12.0,
        randomize_spawn: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(xml_path=xml_path or LUNAR_SCENE, **kwargs)
        if self.terrain_geom_id < 0:
            raise ValueError("月面场景缺少名为 'lunar_terrain' 的 hfield geom")
        if self.model.geom_type[self.terrain_geom_id] != mujoco.mjtGeom.mjGEOM_HFIELD:
            raise ValueError("'lunar_terrain' 不是 hfield —— Go2LunarEnv 需要高度场场景")
        self.spawn_radius = float(spawn_radius)
        self.max_spawn_slope_deg = float(max_spawn_slope_deg)
        self.randomize_spawn = bool(randomize_spawn)
        self.spawn_clearance = self.stand_height  # 站高 0.38（平台地表=0 时复刻关键帧）
        self._ground_z = 0.0  # 缓存，兜底射线未命中

    # ---- 地形相对高度：覆写基类钩子 ----
    def _terrain_height(self, x: float, y: float) -> float:
        z = ray_terrain_height(self.model, self.data, x, y, self.terrain_geom_id)
        return self._ground_z if z is None else z

    def _ground_height(self) -> float:
        z = ray_terrain_height(self.model, self.data,
                               float(self.data.qpos[0]), float(self.data.qpos[1]),
                               self.terrain_geom_id)
        if z is not None:
            self._ground_z = z
        return self._ground_z

    def _local_slope_deg(self, x: float, y: float, d: float = 0.15) -> float:
        zc = self._terrain_height(x, y)
        zx = (ray_terrain_height(self.model, self.data, x + d, y, self.terrain_geom_id)
              or zc) - (ray_terrain_height(self.model, self.data, x - d, y, self.terrain_geom_id) or zc)
        zy = (ray_terrain_height(self.model, self.data, x, y + d, self.terrain_geom_id)
              or zc) - (ray_terrain_height(self.model, self.data, x, y - d, self.terrain_geom_id) or zc)
        return float(np.degrees(np.arctan(np.hypot(zx, zy) / (2.0 * d))))

    def _sample_spawn(self) -> tuple[float, float]:
        """在 spawn_radius 内的圆盘上均匀采样，拒绝坡度过陡处（出生即站平缓地，
        再自行走入起伏/坑/坡）。采样失败回退到原点平台。"""
        for _ in range(40):
            r = self.spawn_radius * np.sqrt(self.np_random.uniform())
            th = self.np_random.uniform(0.0, 2.0 * np.pi)
            x, y = float(r * np.cos(th)), float(r * np.sin(th))
            if self._local_slope_deg(x, y) <= self.max_spawn_slope_deg:
                return x, y
        return 0.0, 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)  # 关键帧复位 + 关节/qvel 噪声 + 采样指令
        if options and "spawn" in options:
            x, y = float(options["spawn"][0]), float(options["spawn"][1])
        elif self.randomize_spawn:
            x, y = self._sample_spawn()
        else:
            x, y = 0.0, 0.0
        gz = self._terrain_height(x, y)
        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = gz + self.spawn_clearance
        mujoco.mj_forward(self.model, self.data)
        self._ground_z = gz
        return self._get_obs(), self._get_info()
