import numpy as np

npz_path = r"data\raw_data\Lille1\Lille_1_vox0.10_B64_S48_log_q95_dens_sat.npz"
data = np.load(npz_path, allow_pickle=False)

VOX = float(data["VOX"][0])
ORIGIN = data["ORIGIN"].astype(np.float64)      # (3,)
BLOCK = int(data["BLOCK"][0])
STRIDE = int(data["STRIDE"][0])
QREF = int(data["QREF"][0])
DENS_QUANT = int(data["DENS_QUANT"][0])
KREFS = data["KREFS"].astype(np.float32)         # (P,)
ROOTS = data["PATCH_ORIGS"].astype(np.int32)     # (P,3)

P = KREFS.shape[0]
print("patches =", P)
if P == 0:
    print("没有块通过筛选（看看 VOX/OCC_MIN/OCC_MAX 是否太苛刻）")
else:
    p = 0  # 看第0个块
    Pp = data[f"P{p}"]                           # (n_occ, 5) uint16
    ijk_rel = Pp[:, :3].astype(np.int32)         # (n_occ, 3)
    dens_q = Pp[:, 3].astype(np.uint16)          # (n_occ,)
    sat    = Pp[:, 4].astype(np.uint16)          # (n_occ,)

    ijk_abs = ijk_rel + ROOTS[p]                 # 全局体素坐标
    xyz_corner = ORIGIN + VOX * ijk_abs          # 世界坐标（体素角点）
    xyz_center = ORIGIN + VOX * (ijk_abs + 0.5)  # 世界坐标（体素中心）

    # 反量化计数（估计值）
    kref = float(KREFS[p])
    d_hat = dens_q / DENS_QUANT
    cnt_hat = np.expm1(d_hat * np.log1p(kref))   # 对数逆变换
    print("n_occ =", ijk_rel.shape[0], "kref =", kref, "sat_ratio =", float((sat>0).mean()))
