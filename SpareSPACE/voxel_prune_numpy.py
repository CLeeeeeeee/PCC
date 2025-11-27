# 用法：
#   python voxel_prune_numpy.py F:\PointCloudPython\data\raw_data\Lille1\Lille_0.ply --voxel 0.20 --method centroid
#   python voxel_prune_numpy.py F:\PointCloudPython\data\raw_data\Lille1\Lille_0.ply --voxel 0.20 --method medoid
#
# 依赖：pip install numpy plyfile open3d

import argparse, time
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
import open3d as o3d

def read_ply_xyzI(path: Path):
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = set(v.dtype.names)
    def pick(*cands):
        for k in cands:
            if k in names: return k
        return None
    kx = pick("x","X","x_coord"); ky = pick("y","Y","y_coord"); kz = pick("z","Z","z_coord")
    assert kx and ky and kz, "未找到坐标字段（x/y/z 或 X/Y/Z）"
    pts = np.vstack([v[kx], v[ky], v[kz]]).T.astype(np.float64)

    ki = pick("intensity","Intensity","reflectance","Reflectance","remission","Remission")
    I = np.asarray(v[ki], dtype=np.float64) if ki else None
    return pts, I

def voxel_group_indices(pts, voxel, origin=None):
    if origin is None:
        origin = pts.min(axis=0)
    keys = np.floor((pts - origin) / voxel).astype(np.int64)   # (N,3) 体素坐标
    # 把 (N,3) 键转为结构化数组，便于唯一化/排序
    key_view = np.core.records.fromarrays(keys.T, names='vx,vy,vz', formats='i8,i8,i8')
    uniq, inv, counts = np.unique(key_view, return_inverse=True, return_counts=True)
    return inv, counts, origin, uniq

def reduce_centroid(pts, inv, counts, I=None):
    # 按组求和（段归约）：sum_x = bincount(inv, weights=pts[:,0])，其他同理
    K = counts.size
    sx = np.bincount(inv, weights=pts[:,0], minlength=K)
    sy = np.bincount(inv, weights=pts[:,1], minlength=K)
    sz = np.bincount(inv, weights=pts[:,2], minlength=K)
    centroids = np.stack([sx, sy, sz], axis=1) / counts[:,None]
    if I is not None:
        sI = np.bincount(inv, weights=I, minlength=K)
        Imean = sI / counts
    else:
        Imean = None
    return centroids.astype(np.float32), (Imean.astype(np.float32) if I is not None else None)

def reduce_medoid(pts, inv, counts, I=None):
    # 先用“组均值”得到每组中心，再为每个点计算到其组中心的距离，
    # 然后按 (inv, dist2) 联合排序，取每组第一个（最小距离）作为 medoid
    cent, _ = reduce_centroid(pts, inv, counts, None)
    dist2 = ((pts - cent[inv])**2).sum(axis=1)
    order = np.lexsort((dist2, inv))            # 先按 inv，再按 dist2
    inv_sorted = inv[order]
    # 找每组的起始位置（变化边界）
    start = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    chosen_idx = order[start]                   # 每组距离最小的点的原始索引
    med = pts[chosen_idx].astype(np.float32)
    Irep = (I[chosen_idx].astype(np.float32) if I is not None else None)
    return med, Irep

def write_ply_xyzI(path: Path, pts, I=None):
    N = pts.shape[0]
    elem = [
        ('x', np.float32), ('y', np.float32), ('z', np.float32)
    ]
    data = np.empty(N, dtype=elem + ([('intensity', np.float32)] if I is not None else []))
    data['x'] = pts[:,0]; data['y'] = pts[:,1]; data['z'] = pts[:,2]
    if I is not None: data['intensity'] = I
    PlyData([PlyElement.describe(data, 'vertex')], text=True).write(str(path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=str)
    ap.add_argument("--voxel", type=float, default=0.20, help="体素边长（米）")
    ap.add_argument("--method", type=str, default="centroid", choices=["centroid","medoid"])
    ap.add_argument("--preview", action="store_true", help="预览结果（Open3D）")
    args = ap.parse_args()

    src = Path(args.src)
    dst = src.with_name(f"{src.stem}_voxel{args.voxel:.2f}_{args.method}.ply")

    t0 = time.time()
    pts, I = read_ply_xyzI(src)
    print(f"[INFO] 读取：{pts.shape[0]:,} 点  hasI={I is not None}")

    t1 = time.time()
    inv, counts, origin, uniq = voxel_group_indices(pts, args.voxel, origin=None)
    print(f"[INFO] 非空体素：{counts.size:,} （压缩比≈ {pts.shape[0]/counts.size:.2f}x）  用时 {time.time()-t1:.2f}s")

    t2 = time.time()
    if args.method == "centroid":
        rep, Irep = reduce_centroid(pts, inv, counts, I)
    else:
        rep, Irep = reduce_medoid(pts, inv, counts, I)
    print(f"[INFO] 代表点生成：{rep.shape[0]:,}  用时 {time.time()-t2:.2f}s  总耗时 {time.time()-t0:.2f}s")

    write_ply_xyzI(dst, rep, Irep)
    print(f"[DONE] 写出：{dst}")

    if args.preview:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rep)
        if Irep is not None:
            I0 = Irep.copy()
            lo, hi = np.percentile(I0, [1,99])
            I0 = np.clip((I0 - lo)/max(hi-lo,1e-6), 0, 1)
            cols = np.stack([I0, I0, I0], 1)
            pcd.colors = o3d.utility.Vector3dVector(cols)
        else:
            pcd.paint_uniform_color([0.2,0.4,0.8])
        pcd = pcd.voxel_down_sample(max(args.voxel, 0.10))  # 预览再轻量点
        o3d.visualization.draw_geometries([pcd], window_name="Voxel Pruned", width=1280, height=800)

if __name__ == "__main__":
    main()
