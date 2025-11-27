# file: quick_check_one_patch.py
import numpy as np
import random

npz_path = r"F:\PointCloudPython\data\raw_data\Lille1\Lille_1_vox0.10_B64_S48_log_q95_dens_sat.npz"
data = np.load(npz_path, allow_pickle=False)

VOX = float(data["VOX"][0])
ORIGIN = data["ORIGIN"].astype(np.float64)
DQUANT = int(data["DENS_QUANT"][0])
ROOTS = data["PATCH_ORIGS"].astype(np.int32)
KREFS = data["KREFS"].astype(np.float32)

P = KREFS.shape[0]
p = random.randrange(P)  # 随机挑一块
arr = data[f"P{p}"]      # (n_occ,5) uint16

ijk_rel = arr[:, :3].astype(np.int32)
dens_q  = arr[:, 3].astype(np.uint16)
sat     = arr[:, 4].astype(np.uint16)

ijk_abs = ijk_rel + ROOTS[p]
xyz_center = ORIGIN + VOX * (ijk_abs + 0.5)

# 反量化估计计数（仅作参考）
kref = float(KREFS[p])
d_hat = dens_q / DQUANT
cnt_hat = np.expm1(d_hat * np.log1p(kref))

print(f"patch #{p}  n_occ={len(arr)}  kref={kref:.3f}  sat%={(sat>0).mean():.3f}")
print("xyz(min)=", xyz_center.min(axis=0), " xyz(max)=", xyz_center.max(axis=0))

# 简单可视化（需 matplotlib）
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    s = (cnt_hat / (cnt_hat.max() + 1e-6)) * 10 + 1  # 点大小随估计计数变化
    ax.scatter(xyz_center[:,0], xyz_center[:,1], xyz_center[:,2], s=s)
    ax.set_title(f"Patch #{p} (n={len(arr)})")
    plt.show()
except Exception as e:
    print("可视化失败(可忽略)：", e)
