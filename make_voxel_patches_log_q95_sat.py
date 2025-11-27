# 文件：make_voxel_patches_log_q95_sat.py
# 运行：python make_voxel_patches_log_q95_sat.py F:\PointCloudPython\data\raw_data\Lille1\Lille_1.ply
# 依赖：pip install numpy plyfile
import sys, numpy as np
from pathlib import Path
from plyfile import PlyData

# ====== 参数（可按需改） ======
VOX = 0.10           # 体素大小（米）
BLOCK = 64           # 块边长（体素）
STRIDE = 48          # 滑窗步长（体素）
OCC_MIN, OCC_MAX = 0.001, 0.10   # 占据率筛选区间
QREF = 95            # K_ref 取每块计数的 q95
DENS_QUANT = 255     # 密度量化到 0..255（uint8）

def read_pts_ply(path: Path):
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = set(v.dtype.names)
    def pick(*c):
        for k in c:
            if k in names: return k
        return None
    kx,ky,kz = pick("x","X","x_coord"), pick("y","Y","y_coord"), pick("z","Z","z_coord")
    assert kx and ky and kz, "未找到坐标字段（x/y/z 或 X/Y/Z）"
    pts = np.vstack([v[kx], v[ky], v[kz]]).T.astype(np.float64)
    return pts
# 只读坐标，不读颜色/强度，可以自行扩展

def voxelize_counts(pts, vox):
    origin = pts.min(axis=0)
    ijk = np.floor((pts - origin)/vox).astype(np.int64)
    ijk = ijk - ijk.min(axis=0)   # 移到非负
    imax, jmax, kmax = ijk.max(axis=0) + 1
    key = ijk[:,0]*(jmax*kmax) + ijk[:,1]*kmax + ijk[:,2]   # 一维化
    uniq, counts = np.unique(key, return_counts=True)
    # uniq是一维排序 双射 counts是对应计数
    vz = uniq % kmax
    vy = (uniq // kmax) % jmax
    vx = uniq // (jmax*kmax)
    vox_idx = np.stack([vx,vy,vz], 1).astype(np.int32)   # 三轴索引转(M,3)数组
    return vox_idx, counts.astype(np.int32), origin, (imax,jmax,kmax)
# 格栅化并计数，也就是点云落在体素上 返回非空体素索引，对应计数，原始坐标系原点，体素维度

def make_patches(vox_idx, counts, dims, block=64, stride=48,
                 occ_min=0.001, occ_max=0.10, qref=95, dens_quant=255):
    # 在体素网格上用三维滑窗生成许多立方块，筛掉太稀/太密的块，对块内非空体素的点数做对数缩放与量化，并记录溢出位。
    imax,jmax,kmax = dims
    I,J,K = vox_idx[:,0], vox_idx[:,1], vox_idx[:,2]
    patches, krefs, roots = [], [], []
    total_skipped_space = 0
    for x0 in range(0, max(1, imax - block + 1), stride):
        x1 = x0 + block
        sel_x = (I>=x0)&(I<x1)
        if not np.any(sel_x): 
            continue
        for y0 in range(0, max(1, jmax - block + 1), stride):
            y1 = y0 + block
            sel_xy = sel_x & (J>=y0)&(J<y1)
            if not np.any(sel_xy): 
                continue
            for z0 in range(0, max(1, kmax - block + 1), stride):
                z1 = z0 + block
                sel = sel_xy & (K>=z0)&(K<z1)
                if not np.any(sel): 
                    total_skipped_space += 1
                    continue
                # 找到非空体素
                idxs = np.nonzero(sel)[0]   # vox_idx 中落入本块的非空体素的行号
                n_occ = idxs.size
                occ_ratio = n_occ / (block**3)
                if occ_ratio < occ_min or occ_ratio > occ_max:
                    continue
                # 筛

                # 块内局部坐标与计数
                loc = vox_idx[idxs] - np.array([x0,y0,z0], dtype=np.int32)
                cnt = counts[idxs].astype(np.float32)

                # 本块的 K_ref（q95，至少为1）
                kref = max(1.0, float(np.percentile(cnt, qref)))
                log1_kref = np.log1p(kref)

                # 对数缩放 + 溢出位
                d = np.log1p(cnt) / log1_kref        # 可能 >1
                sat = (d > 1.0).astype(np.uint8)     # 标记溢出
                d = np.clip(d, 0.0, 1.0)
                dens_q = np.round(d * dens_quant).astype(np.uint8)   
                # 量化到 0..dens_quant 便于存储、训练，后续反量化即可

                # 拼为 [i,j,k,dens_q,sat]（都装进 uint16 便于npz）
                arr = np.zeros((n_occ, 5), dtype=np.uint16)
                arr[:,0:3] = loc.astype(np.uint16)
                arr[:,3] = dens_q.astype(np.uint16)
                arr[:,4] = sat.astype(np.uint16)

                patches.append(arr)
                krefs.append(kref)
                roots.append(np.array([x0,y0,z0], dtype=np.int32))
                # 记录块内相对坐标、kref、本块原点（体素坐标系）
                
    return patches, np.array(krefs, dtype=np.float32), np.vstack(roots).astype(np.int32)

def main():
    src = Path(sys.argv[1]) if len(sys.argv)>1 else None
    assert src and src.exists(), "请提供 PLY 路径"
    print(f"[INFO] 读取: {src}")
    pts = read_pts_ply(src)   # 获取点坐标
    vox_idx, counts, origin, dims = voxelize_counts(pts, VOX)
    print(f"[INFO] 非空体素={vox_idx.shape[0]:,} | 体素维度估计={dims}")
    # 获取全局体素索引、计数、原点、三维

    patches, krefs, roots = make_patches(
        vox_idx, counts, dims,
        block=BLOCK, stride=STRIDE,
        occ_min=OCC_MIN, occ_max=OCC_MAX,
        qref=QREF, dens_quant=DENS_QUANT
    )
    out = src.with_name(f"{src.stem}_vox{VOX:.2f}_B{BLOCK}_S{STRIDE}_log_q{QREF}_dens_sat.npz")
    # patches为(n_occ,5)，[i,j,k,dens_q,sat]; krefs为每块kref; roots为每块原点（体素坐标系）

    meta = {
        "VOX": np.array([VOX], dtype=np.float32),
        "ORIGIN": origin.astype(np.float32),
        "BLOCK": np.array([BLOCK], dtype=np.int32),
        "STRIDE": np.array([STRIDE], dtype=np.int32),
        "QREF": np.array([QREF], dtype=np.int32),
        "DENS_QUANT": np.array([DENS_QUANT], dtype=np.int32),
        "FORMAT": np.array([3], dtype=np.int32),  # 3: 列含义 i,j,k,dens_q(0..255),sat(0/1)
        "KREFS": krefs.astype(np.float32),
        "PATCH_ORIGS": roots.astype(np.int32)
    }   # 记录元数据 ijk_abs = PATCH_ORIGS[p] + Pp[:, :3] → xyz = ORIGIN + VOX * ijk_abs
    for i, arr in enumerate(patches):
        meta[f"P{i}"] = arr
    np.savez_compressed(out, **meta)
    print(f"[DONE] 总块数={len(patches)} | 保存: {out}")

if __name__ == "__main__":
    main()
