# 用法示例：
#   python voxel_prune_with_count.py F:\PointCloudPython\data\raw_data\Lille1\Lille_0.ply --voxel 0.20 --method centroid
# 依赖：pip install numpy plyfile open3d

import argparse, time, numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

def read_ply_xyzI(path: Path):
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

def voxel_group(pts, voxel, origin=None):
    if origin is None:
        origin = pts.min(axis=0)
    keys = np.floor((pts - origin) / voxel).astype(np.int64)           # (N,3)
    key_view = np.core.records.fromarrays(keys.T, names='vx,vy,vz', formats='i8,i8,i8')
    uniq, inv, counts = np.unique(key_view, return_inverse=True, return_counts=True)
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

def medoid_reduce(pts, inv, counts, feat=None):
    # 先取每组质心，再选距质心最近的原始点作为代表
    cent, _ = centroid_reduce(pts, inv, counts, None)
    d2 = ((pts - cent[inv])**2).sum(axis=1)
    order = np.lexsort((d2, inv))
    inv_sorted = inv[order]
    start = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    idx = order[start]
    rep = pts[idx].astype(np.float32)
    f = (feat[idx].astype(np.float32) if feat is not None else None)
    return rep, f

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=str)
    ap.add_argument("--voxel", type=float, default=0.20, help="体素边长（米）")
    ap.add_argument("--method", type=str, default="centroid", choices=["centroid","medoid"])
    args = ap.parse_args()

    src = Path(args.src)
    dst = src.with_name(f"{src.stem}_voxel{args.voxel:.2f}_{args.method}_with_count.ply")

    t0 = time.time()
    pts, I = read_ply_xyzI(src)
    inv, counts, origin, uniq = voxel_group(pts, args.voxel, origin=None)

    if args.method == "centroid":
        rep, Irep = centroid_reduce(pts, inv, counts, I)
    else:
        rep, Irep = medoid_reduce(pts, inv, counts, I)

    write_ply_with_count(
        dst, rep, counts, Irep,
        meta={"origin": origin.tolist(), "voxel_size_m": args.voxel}
    )
    print(f"[DONE] 写出：{dst}")
    print(f"[STATS] 原始点数={pts.shape[0]:,}  剪枝后点数={rep.shape[0]:,}  压缩比≈{pts.shape[0]/rep.shape[0]:.2f}x  总耗时 {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
