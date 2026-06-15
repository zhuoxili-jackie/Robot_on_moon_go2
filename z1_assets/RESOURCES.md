# Z1 机械臂资源清单（Phase 3）

> Phase 3 = 把 Unitree **Z1** 机械臂作为**配重 / payload** 装到 Go2 base 上，平地重训 locomotion。
> 本目录 `z1_assets/` 用于 vendored Z1 网格（把要用的 `.obj` 拷进来，按 ASCII 相对路径在 `go2_z1.xml` 引用，避中文路径坑）。

## 一、本地已有 Z1 资产（无需联网即可起步）

**Z1 网格（9 个 `.obj`，已 .dae→.obj，Felix 转好）**——两处各一份，任拷一份进本目录：
- `../../aliengo_z1_mujoco_scene/aliengo_z1_mujoco_scene/z1_assets/*.obj`
- `../../TRAINING-Aliengo/z1_assets/*.obj`
- 文件：`z1_Link00..06.obj`、`z1_GripperStator.obj`、`z1_GripperMover.obj`

**Z1 运动学 + 惯量（焊接版，直接可抄）**：`../../TRAINING-Aliengo/aliengo_scene.xml` 里已经有一棵完整的 Z1 body 树
（`z1_link00→link01→…→link06→gripper_stator→gripper_mover`），**焊在 aliengo trunk 上、无 `<joint>`、无 `<actuator>` = 纯配重**，
每个 link 带 `inertial`（质量+惯量，总 ~5 kg）。Phase 3 配重方案直接把这棵树移植到 go2 `base_link`（见 CONTINUATION §二）。

## 二、官方 / 干净模型源（留给 Phase 3b「控臂」loco-manipulation）

| 资源 | 内容 | 链接 | 备注 |
|---|---|---|---|
| **mujoco_menagerie / unitree_z1** ★ | 官方维护的 **Z1 MJCF**：6 关节 + **位置执行器** + scene.xml | https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_z1 | 要**控制**手臂（抓矿石/loco-manip）时用这套；也可拿它选一组折叠关节角再焊死当配重 |
| unitree_ros / z1_description | Z1 官方 **URDF** + meshes（.dae/.stl） | https://github.com/unitreerobotics/unitree_ros/tree/master/robots/z1_description | menagerie 的 URDF 源头；Felix 当初也是从 unitree_ros 拿模型转 obj（见根 README「场景说明（Felix）」） |

> **配重 vs 控臂**：Phase 3（本步）只把 Z1 当**死配重**焊在 go2 上重训行走，**用本地焊接版就够**（零关节、env 几乎不改）。
> 真要**控制手臂**（Phase 3b loco-manipulation：抓 metadata 里的 12 个矿石 nodule 等）再上 menagerie 的带关节 MJCF。

## 三、整合要点（详见 CONTINUATION §二 Phase 3 计划）

- **挂载**：z1 链作为 go2 `base_link` 子 body；go2 base 碰撞盒 half=`0.1881×0.04675×0.057`→顶面≈base 系 +0.057 m；
  建议挂**顶面中心、略偏后**，**别照抄 aliengo 的 +0.2535 前移**（go2 base 小，前移过大会栽头）。
- **质量**：Z1 ~4.3–5 kg ≈ go2 整机(~15 kg)的 ~30%——显著配重，CoM 上移/前移，需重训 + 可能微调 stand 姿。
- **臂姿态（最关键）**：把臂折成紧凑「收纳位」让 CoM 尽量在 base 正上方（别用 Felix 的前伸位）。
- **碰撞**：臂在高处当纯配重 → 碰撞 geom 设 `contype=0 conaffinity=0`（不碰地、不自碰，最省心）。
