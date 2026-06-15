# 月面地形资源清单（Phase 2）

> 给 Phase 2 月面迁移用的高度图 / 贴图 / 参考仓库收藏。**本目录 `lunar_assets/` 用于 vendored 月面资产**
> （把要用的 png 拷进来，按 ASCII 相对路径在 `go2_lunar_scene.xml` 里引用，规避中文路径坑）。
> 当前项目自带的月面资产在 `../../lunar_terrain_only/`（见文末「本地已有资产」）。

## 一、真实月面高度图 / 贴图（要「好看 + 真实」就用这些）

| 资源 | 内容 | 链接 | 备注 |
|---|---|---|---|
| **NASA CGI Moon Kit（SVS）** ★首选 | 全月 **displacement（高度）+ color（反照率）+ normal** 贴图，4/16/64 px/度 | https://svs.gsfc.nasa.gov/4720 | NASA 为 CG 艺术家做的，直接给高度位移图+彩色贴图，最适合「好看」。16-bit TIFF（半米/像素，相对半径 1,747,400 m） |
| USGS Astrogeology — LRO LOLA DEM 118m | 全月地形 DEM（118 m/像素） | https://astrogeology.usgs.gov/search/details/Moon/LRO/LOLA/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014/cub | 权威 DEM，需裁剪/降采样成 hfield png |
| PGDA — SLDEM2015 高分辨率地形 | 更高分辨率局部地形 | https://pgda.gsfc.nasa.gov/products/54 | 适合要细节的局部坑/坡 |
| LOLA PDS 数据节点（MIT imbrium） | LOLA 原始测高数据归档 | https://imbrium.mit.edu/ | 原始数据源 |
| sbcode「Moon Heightmap」教程 | 怎么把月面高度图做成 3D | https://sbcode.net/topoearth/moon-heightmap/ | 实操向，含从 NASA 数据生成 heightmap 的流程 |

> **用法**：下载 displacement/DEM → 裁一块平坦区域 → 缩放到 513×513（或 mujoco hfield 任意 nrow/ncol）→ 存 16-bit 或 8-bit png →
> 在 `<hfield file="..." size="rx ry zmax base">` 里用。**`zmax` 直接线性决定起伏幅度**（见 CONTINUATION §2.2「压幅度」）。

## 二、参考仓库 / 论文（地形生成 + Go2 运控 + 键盘遥控）

| 仓库 / 论文 | 与本项目的关系 | 链接 |
|---|---|---|
| **darshmenon/quadruped-dog-rl** ★最相关 | Unitree **Go2** + **MuJoCo** + PPO，**多地形** + **键盘遥控**——几乎覆盖我们 Phase 2+Phase 4 | https://github.com/darshmenon/quadruped-dog-rl |
| google-deepmind/mujoco_playground | 官方 GPU MuJoCo 运控库，含 quadruped + 地形 / curriculum / domain-rand 范式 | https://github.com/google-deepmind/mujoco_playground |
| JewelryForge/QuadrupedRL | 四足运控环境，含「从高程图建地形」接口 | https://github.com/JewelryForge/QuadrupedRL |
| awesome-legged-locomotion-learning | 腿足运控学习资源合集 | https://github.com/gaiyi7788/awesome-legged-locomotion-learning |
| 论文 2010.11251《Learning Quadrupedal Locomotion over Challenging Terrain》 | **Perlin 噪声程序化地形**（roughness/frequency/amplitude 三参数）——可不靠真实 DEM 自造温和起伏 | https://arxiv.org/pdf/2010.11251 |

> ⚠️ **MJX（GPU 分支）不支持 hfield 碰撞**（github.com/google-deepmind/mujoco/issues/1491）。我们用 **CPU MuJoCo**，hfield 碰撞正常，不受影响。

## 三、本地已有资产（`lunar_terrain_only/`）

项目自带一套「中央陨石坑」月面（任坤提供），无需联网即可起步：

- `lunar_heightfield_center_crater.png`（513×513，MuJoCo hfield 源）
- `lunar_albedo_center_crater.png`（反照率贴图）
- `lunar_heightfield_raw_meters.npy` / `_normalized.npy`（高度采样）
- `scene_metadata.json`（地形元数据）、`terrain_preview_center_crater.png`（预览）

**关键参数**（来自 metadata）：hfield `size=[20,20,1.65,0.12]`；实测高度 min −0.55 / max +0.22 / **std 0.072 m**（多数地表 ±7 cm 温和起伏）；
**中央陨石坑 0.62 m 深、半径 4.8 m、在原点 (0,0)**；另有 4 个小坑（边缘）。**起步先用这套**：压低 `zmax` + spawn 避开中央坑（见 CONTINUATION §2.5）。
要更「好看」再换第一节的 NASA CGI Moon Kit。
