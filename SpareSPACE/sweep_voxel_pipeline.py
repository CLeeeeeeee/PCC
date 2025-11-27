# 文件名：sweep_voxel_pipeline.py
# 功能：对多种体素尺寸进行“剪枝→解压→评估”全流程扫描，并导出 CSV。
# python sweep_voxel_pipeline.py --src F:\PointCloudPython\data\raw_data\Lille1\Lille_0.ply --voxels 0.10 0.15 0.20 0.30 --eval_voxel 0.10 --max_src 3000000 --max_rec 3000000 --out_dir F:\PointCloudPython\data\raw_data\Lille1\sweep --csv results_voxel_sweep.csv
# 依赖：pip install numpy scipy open3d plyfile

import argparse, csv, os, time, re
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
from scipy.spatial import cKDTree

# ---------- 基础IO ----------
def read_ply_xyzI(path: Path):
    """兼容 X/Y/Z 大写字段；可返回 intensity（若存在）"""
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = set(v.dtype.names)
    def pick(*c):
        for k in c:
            if k in names: return k
        return None
    kx = pick("x","X","x_coord"); ky = pick("y","Y","y_coord"); kz = pick("z","Z","z_coord")
    assert kx and ky and kz, "未找到坐标字段（x/y/z 或 X/Y/Z）"
    pts = np.vstack([v[kx], v[ky], v[kz]]).T.astype(np.float64)
    ki  = pick("intensity","Intensity","reflectance","Reflectance","remission","Remission")
    I   = np.asarray(v[ki], dtype=np.float32) if ki else None
    return pts, I

def write_ply_with_count(out_path: Path, rep, counts, intensity=None, meta=None):
    N = rep.shape[0]
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('count', np.uint32)]
    if intensity is not None:
        dtype.append(('intensity', np.float32))
    data = np.empty(N, dtype=dtype)
    data['x'], data['y'], data['z'] = rep[:,0], rep[:,1], rep[:,2]
    data['count'] = counts.astype(np.uint32)
    if intensity is not None:
        data['intensity'] = intensity
    ply = PlyData([PlyElement.describe(data, 'vertex')], text=True)
    if meta:
        for k,v in meta.items():
            ply.comments.append(f"{k}: {v}")
    ply.write(str(out_path))

def write_ply_xyzI(path: Path, pts, intensity=None):
    N = pts.shape[0]
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    if intensity is not None:
        dtype.append(('intensity', np.float32))
    data = np.empty(N, dtype=dtype)
    data['x'], data['y'], data['z'] = pts[:,0], pts[:,1], pts[:,2]
    if intensity is not None:
        data['intensity'] = intensity
    PlyData([PlyElement.describe(data,'vertex')], text=True).write(str(path))

# ---------- 体素分组（NumPy 唯一化/段归约） ----------
def voxel_group(pts, voxel, origin=None):
    if origin is None:
        origin = pts.min(axis=0)
    keys = np.floor((pts - origin) / voxel).astype(np.int64)  # (N,3)
    view = np.core.records.fromarrays(keys.T, names='vx,vy,vz', formats='i8,i8,i8')
    uniq, inv, counts = np.unique(view, return_inverse=True, return_counts=True)
    return inv, counts, origin, uniq

def centroid_reduce(pts, inv, counts, feat=None):
    K = counts.size
    sx = np.bincount(inv, weights=pts[:,0], minlength=K)
    sy = np.bincount(inv, weights=pts[:,1], minlength=K)
    sz = np.bincount(inv, weights=pts[:,2], minlength=K)
    rep = np.stack([sx,sy,sz],1) / counts[:,None]
    if feat is not None:
        sf = np.bincount(inv, weights=feat, minlength=K)
        f  = sf / counts
    else:
        f = None
    return rep.astype(np.float32), (f.astype(np.float32) if f is not None else None)

# ---------- 解压（L1：均匀+Halton+抖动） ----------
def _halton(n, base):
    f, r = 1.0, 0.0
    while n > 0:
        f /= base; r += f * (n % base); n //= base
    return r
def halton_3d(N, start=1):
    idx = np.arange(start, start+N, dtype=np.int64)
    v = np.vectorize
    return np.stack([v(_halton)(idx,2), v(_halton)(idx,3), v(_halton)(idx,5)], axis=1)

def decompress_uniform(rep_pts, counts, origin, voxel_size, intensity=None, jitter_scale=0.2, cap_per_voxel=None):
    vij = np.floor((rep_pts - origin) / voxel_size).astype(np.int64)
    base = origin + vij * voxel_size
    out_pts = []
    out_I   = [] if intensity is not None else None
    seeds = (vij[:,0]*73856093 ^ vij[:,1]*19349663 ^ vij[:,2]*83492791) & 0xFFFFFFFF
    for m, k in enumerate(counts):
        k = int(k)
        if cap_per_voxel is not None:
            k = min(k, cap_per_voxel)
        if k <= 0: 
            continue
        if k == 1:
            out_pts.append(rep_pts[m:m+1])
            if out_I is not None: out_I.append(np.full((1,), intensity[m], dtype=np.float32))
            continue
        h = halton_3d(k, start=int(seeds[m]%100000 + 1))
        rng = np.random.default_rng(seeds[m])
        jitter = (rng.random((k,3)) - 0.5) * jitter_scale
        u = np.clip(h + jitter, 0, 1)
        pts = base[m] + u * voxel_size
        out_pts.append(pts)
        if out_I is not None:
            out_I.append(np.full((k,), intensity[m], dtype=np.float32))
    P = np.vstack(out_pts) if out_pts else np.zeros((0,3), dtype=np.float32)
    I = np.concatenate(out_I) if out_I is not None and out_pts else None
    return P, I

# ---------- 评估（Chamfer 双向） ----------
def voxel_downsample_np(pts, voxel):
    if voxel is None or voxel <= 0: return pts
    origin = pts.min(axis=0)
    keys = np.floor((pts - origin)/voxel).astype(np.int64)
    view = np.core.records.fromarrays(keys.T, names='vx,vy,vz', formats='i8,i8,i8')
    _, inv = np.unique(view, return_inverse=True)
    K = inv.max()+1
    sx = np.bincount(inv, weights=pts[:,0], minlength=K)
    sy = np.bincount(inv, weights=pts[:,1], minlength=K)
    sz = np.bincount(inv, weights=pts[:,2], minlength=K)
    cnt = np.bincount(inv, minlength=K)
    rep = np.stack([sx,sy,sz],1) / cnt[:,None]
    return rep.astype(np.float32)

def nn_dist(a, b):
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)  # 如需并行升级 SciPy 后用: workers=-1
    return d.astype(np.float32)

def eval_chamfer(src_pts, rec_pts, eval_voxel=0.0, max_src=3_000_000, max_rec=3_000_000, seed=0):
    rng = np.random.default_rng(seed)
    A = src_pts
    B = rec_pts
    if A.shape[0] > max_src:
        A = A[rng.choice(A.shape[0], size=max_src, replace=False)]
    if B.shape[0] > max_rec:
        B = B[rng.choice(B.shape[0], size=max_rec, replace=False)]
    if eval_voxel and eval_voxel > 0:
        A = voxel_downsample_np(A, eval_voxel)
        B = voxel_downsample_np(B, eval_voxel)
    dAB = nn_dist(A, B)
    dBA = nn_dist(B, A)
    chamfer_L2_sq = (dAB**2).mean() + (dBA**2).mean()
    chamfer_L2 = np.sqrt(chamfer_L2_sq)
    def pack(name, d):
        return dict(name=name, mean=float(d.mean()), median=float(np.median(d)),
                    p95=float(np.percentile(d,95)), rmse=float(np.sqrt((d**2).mean())), 
                    max=float(d.max()))
    return chamfer_L2, chamfer_L2_sq, pack("A2B", dAB), pack("B2A", dBA), A.shape[0], B.shape[0]

# ---------- 主流程：循环体素 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始 PLY 路径")
    ap.add_argument("--voxels", nargs="+", type=float, default=[0.10,0.15,0.20,0.30])
    ap.add_argument("--method", choices=["centroid","medoid"], default="centroid")  # 这里实现的是centroid
    ap.add_argument("--cap_per_voxel", type=int, default=None, help="解压时每体素最多点数")
    ap.add_argument("--eval_voxel", type=float, default=0.10, help="评估体素(m)")
    ap.add_argument("--max_src", type=int, default=3_000_000)
    ap.add_argument("--max_rec", type=int, default=3_000_000)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--csv", type=str, default="sweep_results.csv")
    args = ap.parse_args()

    src_path = Path(args.src)
    out_dir = Path(args.out_dir) if args.out_dir else src_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取原始点云：{src_path}")
    src_pts, src_I = read_ply_xyzI(src_path)
    src_size = os.path.getsize(src_path) if src_path.exists() else None

    rows = []
    for vx in args.voxels:
        print(f"\n===== Voxel = {vx:.2f} m =====")
        t0 = time.time()

        # 剪枝
        inv, counts, origin, uniq = voxel_group(src_pts, vx, origin=None)
        rep, Irep = centroid_reduce(src_pts, inv, counts, src_I)  # 这里使用 centroid
        pruned_path = out_dir / f"{src_path.stem}_voxel{vx:.2f}_{args.method}_with_count.ply"
        write_ply_with_count(pruned_path, rep, counts, Irep, meta={"origin": origin.tolist(), "voxel_size_m": vx})
        pruned_size = os.path.getsize(pruned_path)
        print(f"[PRUNE] 点数 {src_pts.shape[0]:,} -> {rep.shape[0]:,} （压缩比≈{src_pts.shape[0]/rep.shape[0]:.2f}x） 文件≈{(src_size or 0)/1e6:.1f}MB -> {pruned_size/1e6:.1f}MB")

        # 解压
        recon_pts, Irec = decompress_uniform(rep, counts, origin, vx, intensity=Irep, jitter_scale=0.2, cap_per_voxel=args.cap_per_voxel)
        recon_path = out_dir / f"{src_path.stem}_voxel{vx:.2f}_recon.ply"
        write_ply_xyzI(recon_path, recon_pts, Irec)
        recon_size = os.path.getsize(recon_path)
        print(f"[DECOMP] 重建点数 {recon_pts.shape[0]:,}  文件≈{recon_size/1e6:.1f}MB")

        # 评估
        chamfer, chamfer_sq, A2B, B2A, A_eval_n, B_eval_n = eval_chamfer(
            src_pts, recon_pts, eval_voxel=args.eval_voxel,
            max_src=args.max_src, max_rec=args.max_rec, seed=0
        )
        t1 = time.time()

        print(f"[EVAL] Chamfer(L2,开根)={chamfer:.4f} m | A→B mean={A2B['mean']:.4f} m | B→A mean={B2A['mean']:.4f} m | 用时 {t1-t0:.1f}s")

        rows.append({
            "voxel_m": vx,
            "src_points": src_pts.shape[0],
            "pruned_points": rep.shape[0],
            "recon_points": recon_pts.shape[0],
            "point_compress_ratio": src_pts.shape[0]/rep.shape[0],
            "src_file_mb": (src_size or 0)/1e6,
            "pruned_file_mb": pruned_size/1e6,
            "recon_file_mb": recon_size/1e6,
            "file_ratio_src_to_pruned": (src_size or 1)/max(pruned_size,1),
            "file_ratio_src_to_recon": (src_size or 1)/max(recon_size,1),
            "A_eval_points": A_eval_n,
            "B_eval_points": B_eval_n,
            "A2B_mean_m": A2B["mean"],
            "A2B_median_m": A2B["median"],
            "A2B_p95_m": A2B["p95"],
            "A2B_rmse_m": A2B["rmse"],
            "B2A_mean_m": B2A["mean"],
            "B2A_median_m": B2A["median"],
            "B2A_p95_m": B2A["p95"],
            "B2A_rmse_m": B2A["rmse"],
            "Chamfer_L2_m": chamfer,
            "Chamfer_L2_sq_m2": chamfer_sq,
            "time_s": t1 - t0
        })

    # 写 CSV
    csv_path = out_dir / args.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[DONE] 扫描完成，结果已保存：{csv_path}")
    print("提示：把 voxel_m 作为横坐标、Chamfer_L2_m 或 A2B_mean_m/B2A_mean_m 作为纵坐标画图；")
    print("再配合 point_compress_ratio 或 pruned_file_mb 画“压缩比–误差”曲线。")

if __name__ == "__main__":
    main()
