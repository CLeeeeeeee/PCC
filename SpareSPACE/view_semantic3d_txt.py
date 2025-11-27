# 文件名：view_semantic3d_txt.py
# 用法：python view_semantic3d_txt.py F:\PointCloudPython\data\your_scene.txt
# 依赖：pip install open3d numpy pandas

import sys, numpy as np, pandas as pd, open3d as o3d
from pathlib import Path

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else None
assert PATH and PATH.exists(), "F:\PointCloudPython\data\MarketplaceFeldkirch_Station4_rgb_intensity-reduced.txt"

# 可调：为了顺畅先取前 N 百万点（None=全量）
MAX_POINTS = 3_000_000
VOXEL_SIZE = 0.10    # 10cm 下采样，越大越快
POINT_SIZE = 3.0
USE_AUTO_CLIP = True # 自动分位截断以提升颜色/强度对比度

# -------- 快速探测前几行，判断列数 --------
with open(PATH, "r") as f:
    head = [next(f) for _ in range(10)]
ncols = len(head[0].split())
# 常见：6=xyz rgb；7=xyz i rgb；7/8=含label
# 按位置猜测列义
has_intensity = (ncols >= 7)
has_label = (ncols in (7,8)) and (head[0].split()[-1].isdigit())

# -------- 用 pandas 按块读取，避免一次性吃内存 --------
usecols = list(range(ncols))
colnames = [f"c{i}" for i in usecols]
chunks = []
read_rows = 0

for chunk in pd.read_csv(PATH, sep=r"\s+", header=None, names=colnames,
                         usecols=usecols, engine="python", chunksize=1_000_000):
    chunks.append(chunk)
    read_rows += len(chunk)
    if MAX_POINTS and read_rows >= MAX_POINTS:
        break

df = pd.concat(chunks, ignore_index=True)
df = df.astype(np.float32)

# -------- 解析列：xyz / rgb / intensity（若有）--------
x, y, z = df["c0"], df["c1"], df["c2"]
if has_intensity and ncols >= 7:
    I = df["c3"].to_numpy(dtype=np.float32)
    r, g, b = df["c4"], df["c5"], df["c6"]
else:
    I = None
    r, g, b = df["c3"], df["c4"], df["c5"]

xyz = np.stack([x, y, z], axis=1)

# 颜色归一化为 0..1
rgb = np.stack([r, g, b], axis=1).astype(np.float32)
if rgb.max() > 1.5:  # 0..255 -> 0..1
    rgb /= 255.0
if USE_AUTO_CLIP:
    # 防极值：按分位截断再线性拉伸
    lo, hi = np.percentile(rgb, [1, 99], axis=0)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)

# -------- 组装点云 & 体素下采样 --------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

print(f"[INFO] 加载点数：{np.asarray(pcd.points).shape[0]:,}")
pcd = pcd.voxel_down_sample(VOXEL_SIZE)
print(f"[INFO] 下采样后：{np.asarray(pcd.points).shape[0]:,}  (voxel={VOXEL_SIZE} m)")

# -------- 显示（白底 + 大点）--------
vis = o3d.visualization.Visualizer()
vis.create_window("Semantic3D (XYZRGB)", width=1280, height=800, visible=True)
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.point_size = POINT_SIZE
opt.background_color = np.array([1.0, 1.0, 1.0])
vis.run(); vis.destroy_window()
