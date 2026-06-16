"""程序化生成「可设计」的月面高度场（Phase 2 用）。

为什么自造而不是直接用 lunar_terrain_only 的 center_crater.png：
  1. center_crater 的大坑在**原点 (0,0)**，正好压在出生点 —— spawn 会直接掉进坑。
  2. 它的 zmax=1.65 把 ±7cm 的温和起伏垂直拉伸 ~2.1×，对 0.3m 高的 go2 过陡。
  3. 用户要求「有起伏 + 特定位置有大坑 + 狗能攀爬不摔」—— 需要我**精确控制**：
     哪里平、哪里坑、哪里坡、坡多陡。程序化生成才能拿捏难度（还能做课程）。

设计（世界坐标，单位 m）：
  * 原点平台：半径 PAD_R0 的圆盘恒为 z=0（出生台），PAD_R0..PAD_R1 平滑过渡到起伏。
    —— 这样带臂 go2 的 'stand' 关键帧（base_z=0.38、足端落 z≈0）无需改动即可出生贴地。
  * 温和滚动起伏：多倍频 value-noise（高斯模糊白噪声），峰谷 ±5~8cm、波长数米 → 缓坡，可走。
  * 大坑（craters）：抛物面坑底 + 高斯环形坑沿（rim），放在远离原点的指定位置。
    抛物面壁最陡处坡度 = 2·depth/r，选 depth/r 使壁坡 ≲ 20° —— 即便 0.45m 深的大坑也可攀爬。
  * 可攀爬山丘 / 斜坡：高斯山包，给「攀爬」训练用。

★ MuJoCo hfield 高度语义：world_z(x,y) = geom_pos_z + zmax · png_norm(x,y)，png_norm∈[0,1]。
  令 png_norm = (H − Hmin)/(Hmax − Hmin)、zmax = Hmax − Hmin、geom_pos_z = Hmin
  ⇒ world_z = Hmin + (H − Hmin) = H，即世界地表 == 我设计的 H（原点平台精确落在 world z=0）。
  meta.json 把这些参数写出来，scene xml 与 env 解析式查找都据此对齐。

用法：python lunar_assets/make_lunar_hfield.py    （在 TRAINING-Go2/ 目录内运行，规避中文路径坑）
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

# 中文路径坑：切到脚本目录，用相对 ASCII 路径写文件。
os.chdir(Path(__file__).resolve().parent)

# ----------------------------- 可调参数 -----------------------------
N = 513                      # hfield 分辨率（nrow=ncol）
R = 12.5                     # 半边长（m）；地形覆盖 [-R, R]²，即 25m×25m
PAD_R0 = 2.0                 # 出生平台半径（恒 z=0）
PAD_R1 = 3.3                 # 平台→起伏 的过渡外径
Z_BASE = 0.5                 # hfield 底盒厚度（m），仅影响 underside，给足余量
SEED = 7
FRICTION_NOTE = "1.0 0.02 0.01"   # 仅记录，真正写在 scene xml

# 大坑：(name, cx, cy, r 半径, depth 深, rim 坑沿高, rim_w 坑沿宽)
#   ★ v2 设计原则（狗是「盲走」、obs 无地形高度）：坑要「大但可攀爬」。
#   抛物面壁坡 ≈ 2·depth/r。控制 depth/r ≈ 0.10–0.11 → 壁坡 ~11–12°，盲走的带臂狗能下能上。
#   ——上一版 depth/r≈0.17（壁 19°）+ 高窄坑沿（局部 31°）盲走必摔（1.5M eval 实证）。
#   现在加大半径、压低深度、放缓加宽坑沿：视觉仍是醒目大坑（直径 6–8m），但坡缓可通行。
CRATERS = [
    # 主大坑（用户的「大坑」）：直径 7.6m、深 0.40m，但壁仅 ~12°，可从容下/上
    ("big_ne",   5.8,  4.8, 3.8, 0.40, 0.05, 0.95),
    ("big_w",   -6.8,  1.0, 3.2, 0.32, 0.045, 0.85),
    # 中坑
    ("med_sw",  -5.4, -4.6, 2.6, 0.24, 0.035, 0.70),
    # 小坑（浅，轻松通过）
    ("small_se", 4.8, -5.8, 1.6, 0.13, 0.022, 0.50),
    ("small_n",  0.2,  7.6, 1.4, 0.11, 0.02, 0.45),
]

# 可攀爬山丘 / 斜坡：(name, cx, cy, height, sigma)  高斯山包，最陡坡 ≈ 0.61·height/sigma
#   保持 ≲ 10–12° 的可攀爬范围（盲走能上）。hill_e 上一版 8° 已被验证爬得很好。
HILLS = [
    ("hill_e",   6.8, -1.0, 0.30, 2.4),   # 大缓山包，最陡 ~4°（验证已会爬）
    ("mound_nw",-3.4,  5.6, 0.26, 1.5),   # 稍陡的小包，最陡 ~6°，练攀爬
    ("ridge_climb", 1.6, -7.6, 0.34, 1.7),  # 专门的「爬坡」挑战，最陡 ~7°，仍盲走可上
]
# --------------------------------------------------------------------


def _grid():
    xs = np.linspace(-R, R, N)
    ys = np.linspace(-R, R, N)
    X, Y = np.meshgrid(xs, ys)        # X[iy,ix], Y[iy,ix]；iy→y, ix→x
    return X, Y


def _value_noise(cells: int, smooth: float, rng) -> np.ndarray:
    """白噪声粗网格 → 三次上采样 → 高斯模糊 → 标准化为单位方差，得平滑起伏一倍频。"""
    coarse = rng.standard_normal((cells, cells))
    up = ndimage.zoom(coarse, N / cells, order=3)[:N, :N]
    up = ndimage.gaussian_filter(up, smooth, mode="reflect")
    up = up / (np.std(up) + 1e-9)
    return up


def _rolling_terrain(rng) -> np.ndarray:
    """多倍频温和滚动起伏（fBm 风）。幅度小、波长大 → 缓坡可走。"""
    base = (
        0.055 * _value_noise(8, 6.0, rng)     # 大尺度缓坡
        + 0.025 * _value_noise(18, 3.0, rng)  # 中尺度
        + 0.012 * _value_noise(40, 1.6, rng)  # 细微纹理
    )
    return base


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _crater(X, Y, cx, cy, r, depth, rim, rim_w) -> np.ndarray:
    """抛物面坑 + 高斯环坑沿。返回叠加到地形上的 Δz（坑内为负、沿口为正）。"""
    s = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    u = s / r
    bowl = np.where(u < 1.0, -depth * (1.0 - u ** 2), 0.0)
    ring = rim * np.exp(-((s - r) / rim_w) ** 2)
    return bowl + ring


def _hill(X, Y, cx, cy, height, sigma) -> np.ndarray:
    s2 = (X - cx) ** 2 + (Y - cy) ** 2
    return height * np.exp(-s2 / (2.0 * sigma ** 2))


def _slope_degrees(H: np.ndarray) -> np.ndarray:
    dx = (2.0 * R) / (N - 1)
    gy, gx = np.gradient(H, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def build() -> dict:
    rng = np.random.default_rng(SEED)
    X, Y = _grid()
    s_origin = np.sqrt(X ** 2 + Y ** 2)

    # 1) 滚动起伏，乘以「出生平台抠平」权重（平台内 0，平台外 1）
    pad_w = _smoothstep(PAD_R0, PAD_R1, s_origin)
    H = _rolling_terrain(rng) * pad_w

    # 2) 叠加大坑与山丘（都在平台外）
    for (_n, cx, cy, r, depth, rim, rim_w) in CRATERS:
        H = H + _crater(X, Y, cx, cy, r, depth, rim, rim_w)
    for (_n, cx, cy, h, sg) in HILLS:
        H = H + _hill(X, Y, cx, cy, h, sg)

    # 3) 平台再抠一次（防止坑/丘的尾巴渗进出生台）并轻微整体平滑去毛刺
    H = H * pad_w
    H = ndimage.gaussian_filter(H, 1.0, mode="reflect")
    # 平滑后平台可能微偏，强制平台内精确归零
    H = H * _smoothstep(PAD_R0 * 0.9, PAD_R1, s_origin)

    Hmin, Hmax = float(H.min()), float(H.max())
    zmax = Hmax - Hmin
    geom_pos_z = Hmin                      # world_z = geom_pos_z + zmax·png_norm = H

    # 写 16-bit PNG（红/灰度通道；上下翻转使图像 top 行=+y，MuJoCo 载入再翻回）
    norm = (H - Hmin) / (zmax + 1e-12)
    png16 = np.clip(np.round(norm * 65535.0), 0, 65535).astype(np.uint16)
    _save_png16(np.flipud(png16), "lunar_designed_hfield.png")

    np.save("lunar_designed_heightfield_meters.npy", H.astype(np.float32))

    slope = _slope_degrees(H)
    meta = {
        "resolution": [N, N],
        "half_extent_m": R,
        "world_xy_extent": [-R, R, -R, R],
        "hfield_size": [R, R, round(zmax, 6), Z_BASE],   # 直接写进 <hfield size=...>
        "geom_pos": [0.0, 0.0, round(geom_pos_z, 6)],    # 直接写进 <geom pos=...>
        "spawn_pad": {"r0": PAD_R0, "r1": PAD_R1, "world_z": 0.0},
        "height_stats_m": {"min": round(Hmin, 4), "max": round(Hmax, 4),
                            "std": round(float(H.std()), 4)},
        "slope_stats_deg": {"max": round(float(slope.max()), 1),
                            "mean": round(float(slope.mean()), 2),
                            "p99": round(float(np.percentile(slope, 99)), 1)},
        "craters": [{"name": n, "cx": cx, "cy": cy, "r": r, "depth": depth,
                     "rim": rim, "wall_slope_deg": round(np.degrees(np.arctan(2 * depth / r)), 1)}
                    for (n, cx, cy, r, depth, rim, _w) in CRATERS],
        "hills": [{"name": n, "cx": cx, "cy": cy, "height": h, "sigma": sg,
                   "max_slope_deg": round(np.degrees(np.arctan(0.6065 * h / sg)), 1)}
                  for (n, cx, cy, h, sg) in HILLS],
        "friction_note": FRICTION_NOTE,
        "png_file": "lunar_designed_hfield.png",
        "npy_file": "lunar_designed_heightfield_meters.npy",
    }
    Path("lunar_designed_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    _preview(H, slope, meta)
    return meta


def _save_png16(arr_u16: np.ndarray, path: str) -> None:
    from PIL import Image
    Image.fromarray(arr_u16, mode="I;16").save(path)


def _preview(H, slope, meta) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [-R, R, -R, R]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    # (a) 着色高程 + 等高线 + 坑/丘标注
    ax = axes[0, 0]
    im = ax.imshow(H, origin="lower", extent=extent, cmap="terrain")
    ax.contour(H, levels=14, extent=extent, colors="k", linewidths=0.3, alpha=0.5)
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(PAD_R0 * np.cos(th), PAD_R0 * np.sin(th), "w--", lw=1.2)
    ax.plot(PAD_R1 * np.cos(th), PAD_R1 * np.sin(th), "w:", lw=0.8)
    for c in meta["craters"]:
        ax.plot(c["cx"], c["cy"], "rv", ms=7)
        ax.text(c["cx"], c["cy"] + 0.4, c["name"], color="r", fontsize=7, ha="center")
    for h in meta["hills"]:
        ax.plot(h["cx"], h["cy"], "b^", ms=7)
        ax.text(h["cx"], h["cy"] + 0.4, h["name"], color="b", fontsize=7, ha="center")
    ax.set_title("elevation (m) — white dashed = spawn pad")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (b) 山体阴影（hillshade）看「好不好看」
    ax = axes[0, 1]
    dx = (2 * R) / (N - 1)
    gy, gx = np.gradient(H, dx)
    az, alt = np.radians(315), np.radians(35)
    nx, ny, nz = -gx, -gy, np.ones_like(H)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    shade = (np.cos(alt) * (np.cos(az) * nx + np.sin(az) * ny) / norm + np.sin(alt) * nz / norm)
    ax.imshow(shade, origin="lower", extent=extent, cmap="gray")
    ax.set_title("hillshade (lunar look)")

    # (c) 坡度图（红=陡，看可通行性）
    ax = axes[1, 0]
    im = ax.imshow(slope, origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=30)
    ax.set_title(f"slope (deg)  max={meta['slope_stats_deg']['max']}  p99={meta['slope_stats_deg']['p99']}")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (d) 过原点的两条剖面线
    ax = axes[1, 1]
    mid = N // 2
    xs = np.linspace(-R, R, N)
    ax.plot(xs, H[mid, :], label="cross-section y=0")
    ax.plot(xs, H[:, mid], label="cross-section x=0")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(-PAD_R0, PAD_R0, color="green", alpha=0.12, label="spawn pad")
    ax.set_xlabel("world (m)"); ax.set_ylabel("height (m)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("profiles through origin")

    fig.suptitle("designed lunar heightfield", fontsize=14)
    fig.tight_layout()
    fig.savefig("lunar_designed_preview.png", dpi=95)
    plt.close(fig)


if __name__ == "__main__":
    meta = build()
    print("== designed lunar heightfield ==")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
