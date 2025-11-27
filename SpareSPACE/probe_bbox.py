# 文件名：probe_bbox.py
# 依赖：pip install plyfile numpy
import sys, numpy as np
from pathlib import Path
from plyfile import PlyData

PLY_PATH = Path(sys.argv[1] if len(sys.argv)>1 else r"F:\PointCloudPython\data\Lille_14.ply")
ply = PlyData.read(str(PLY_PATH))
v = ply["vertex"].data
names = set(v.dtype.names)
def pick_name(cands):
    for k in cands:
        if k in names: return k
    return None

kx = pick_name(("x","X","x_coord")); ky = pick_name(("y","Y","y_coord")); kz = pick_name(("z","Z","z_coord"))
assert kx and ky and kz, "坐标字段未找到"

X = np.asarray(v[kx], dtype=np.float64)
Y = np.asarray(v[ky], dtype=np.float64)
Z = np.asarray(v[kz], dtype=np.float64)

def stats(arr, name):
    q1,q50,q99 = np.percentile(arr, [1,50,99])
    print(f"{name}: min={arr.min():.3f}, q1={q1:.3f}, median={q50:.3f}, q99={q99:.3f}, max={arr.max():.3f}")

print(f"[INFO] 点数: {X.size:,}")
stats(X, "X"); stats(Y, "Y"); stats(Z, "Z")
print("[HINT] 建议先选 ROI = [q1,q99] 区间（或稍微扩一点），再做体素下采样。")
