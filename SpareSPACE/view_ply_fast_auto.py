# 文件名：view_ply_fast_auto.py
# 依赖：pip install open3d plyfile numpy

import sys, numpy as np, open3d as o3d
from pathlib import Path
from plyfile import PlyData

PLY = Path(sys.argv[1] if len(sys.argv)>1 else r"F:\PointCloudPython\data\Lille_14.ply")
assert PLY.exists(), f"找不到文件：{PLY}"

# ==== 参数（可改）====
AUTO_ROI = True            # True=用分位数自动裁剪；False=用手动 ROI
MANUAL_ROI = {             # 只有 AUTO_ROI=False 时生效
    "X": (-80, 80),
    "Y": (-80, 80),
    "Z": (-5, 20),
}
VOXEL_SIZE = 0.25          # 体素大小（米），可 0.2~0.5
RAND_PRE_SAMPLE = 6_000_000  # ROI 后如仍>此阈值，先随机抽样这么多点
POINT_SIZE = 3.0

# ==== 读取 ====
ply = PlyData.read(str(PLY))
v = ply["vertex"].data
names = set(v.dtype.names)

def pick_name(cands):
    for k in cands:
        if k in names: return k
    return None

kx = pick_name(("x","X","x_coord")); ky = pick_name(("y","Y","y_coord")); kz = pick_name(("z","Z","z_coord"))
assert kx and ky and kz, "坐标字段未找到"

X = np.asarray(v[kx], dtype=np.float32)
Y = np.asarray(v[ky], dtype=np.float32)
Z = np.asarray(v[kz], dtype=np.float32)
N = X.size
print(f"[INFO] 原始点数: {N:,}")

# ---- 自动/手动 ROI ----
if AUTO_ROI:
    def qrange(arr):
        q1,q99 = np.percentile(arr, [1,99])
        return float(q1), float(q99)
    RX, RY, RZ = qrange(X), qrange(Y), qrange(Z)
else:
    RX, RY, RZ = MANUAL_ROI["X"], MANUAL_ROI["Y"], MANUAL_ROI["Z"]

mask = (X>=RX[0])&(X<=RX[1])&(Y>=RY[0])&(Y<=RY[1])&(Z>=RZ[0])&(Z<=RZ[1])
idx = np.nonzero(mask)[0]
print(f"[INFO] ROI: X{RX}, Y{RY}, Z{RZ} -> 保留 {idx.size:,} 点")

if idx.size == 0:
    print("[WARN] ROI 裁剪后没有点。请放宽 ROI（或设 AUTO_ROI=True）。")
    sys.exit(0)

# ---- ROI 后随机预抽样（防爆内存/卡顿）----
if idx.size > RAND_PRE_SAMPLE:
    sel = np.random.default_rng(0).choice(idx, size=RAND_PRE_SAMPLE, replace=False)
else:
    sel = idx

xyz = np.stack([X[sel], Y[sel], Z[sel]], axis=1)

# ---- 强度（可选）----
I = None
ki = pick_name(("intensity","Intensity","reflectance","Reflectance","remission","Remission"))
if ki:
    Iraw = np.asarray(v[ki], dtype=np.float32)[sel]
    if Iraw.size > 0:
        lo, hi = np.percentile(Iraw, [1, 99])
        I = np.clip((Iraw - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        I = None

# ---- 组装 + 下采样 ----
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
if I is not None:
    gray = np.stack([I, I, I], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(gray)
else:
    pcd.paint_uniform_color([0.2, 0.4, 0.8])

print(f"[INFO] 下采样前: {np.asarray(pcd.points).shape[0]:,}")
pcd = pcd.voxel_down_sample(VOXEL_SIZE)
print(f"[INFO] 下采样后: {np.asarray(pcd.points).shape[0]:,}  (voxel={VOXEL_SIZE} m)")

# ---- 显示（白背景+大点）----
vis = o3d.visualization.Visualizer()
vis.create_window("Paris-Lille-3D (fast auto)", width=1280, height=800, visible=True)
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.point_size = POINT_SIZE
opt.background_color = np.array([1.0, 1.0, 1.0])
vis.run(); vis.destroy_window()
