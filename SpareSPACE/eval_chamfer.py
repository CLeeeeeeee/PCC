# 用法示例：
#   python eval_chamfer.py --src F:\PointCloudPython\data\raw_data\Lille1\Lille_0.ply --rec F:\PointCloudPython\data\raw_data\Lille1\Lille_0_recon.ply --voxel 0.10 --max_src 3000000 --max_rec 3000000
#
# 依赖：pip install numpy scipy open3d plyfile laspy

import argparse, os, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import open3d as o3d

def load_points(path: Path, max_points=None):
    p = str(path)
    if p.lower().endswith((".las",".laz")):
        import laspy
        las = laspy.read(p)
        pts = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
    else:
        # 让 Open3D 读（字段名不规范时可退回 plyfile，但 o3d 对大文件更省心）
        pcd = o3d.io.read_point_cloud(p)
        if len(pcd.points) == 0 and p.lower().endswith(".ply"):
            # 兼容大写 X/Y/Z 的 PLY：用 plyfile 兜底
            from plyfile import PlyData
            v = PlyData.read(p)["vertex"].data
            names = v.dtype.names
            def pick(*c):
                for k in c:
                    if k in names: return k
                return None
            kx,ky,kz = pick("x","X","x_coord"),pick("y","Y","y_coord"),pick("z","Z","z_coord")
            assert kx and ky and kz, "未找到坐标字段"
            pts = np.vstack([v[kx],v[ky],v[kz]]).T.astype(np.float32)
        else:
            pts = np.asarray(pcd.points, dtype=np.float32)
    if (max_points is not None) and (pts.shape[0] > max_points):
        sel = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[sel]
    return pts

def voxel_downsample_np(pts, voxel):
    if voxel is None or voxel <= 0:
        return pts
    origin = pts.min(axis=0)
    keys = np.floor((pts - origin)/voxel).astype(np.int64)
    view = np.core.records.fromarrays(keys.T, names='vx,vy,vz', formats='i8,i8,i8')
    _, inv = np.unique(view, return_inverse=True)
    # 取每组第一个索引（更快）或取质心（更准）。这里用质心：
    K = inv.max()+1
    sx = np.bincount(inv, weights=pts[:,0], minlength=K)
    sy = np.bincount(inv, weights=pts[:,1], minlength=K)
    sz = np.bincount(inv, weights=pts[:,2], minlength=K)
    cnt = np.bincount(inv, minlength=K)
    rep = np.stack([sx,sy,sz],1) / cnt[:,None]
    return rep.astype(np.float32)

def nn_dist(a, b):
    tree = cKDTree(b)
    d2, _ = tree.query(a, k=1)
    return d2.astype(np.float32)  # 实际是距离，不是平方

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始点云 PLY/LAZ/LAS")
    ap.add_argument("--rec", required=True, help="重建点云 PLY/LAZ/LAS")
    ap.add_argument("--voxel", type=float, default=0.0, help="评估前对双方做体素下采样的体素尺(m)，如 0.10")
    ap.add_argument("--max_src", type=int, default=3_000_000, help="原始最大采样点数")
    ap.add_argument("--max_rec", type=int, default=3_000_000, help="重建最大采样点数")
    args = ap.parse_args()

    src, rec = Path(args.src), Path(args.rec)
    src_sz = os.path.getsize(src) if src.exists() else None
    rec_sz = os.path.getsize(rec) if rec.exists() else None

    print("[INFO] 读取中…")
    A = load_points(src, max_points=args.max_src)
    B = load_points(rec, max_points=args.max_rec)
    print(f"[INFO] 原始点数={A.shape[0]:,}  重建点数={B.shape[0]:,}")

    if args.voxel and args.voxel > 0:
        print(f"[INFO] 体素下采样用于评估：voxel={args.voxel} m")
        A = voxel_downsample_np(A, args.voxel)
        B = voxel_downsample_np(B, args.voxel)
        print(f"[INFO] 下采后：原始={A.shape[0]:,}  重建={B.shape[0]:,}")

    print("[INFO] 最近邻距离（A→B）…")
    dAB = nn_dist(A, B)  # 每个A点到B的最近距离
    print("[INFO] 最近邻距离（B→A）…")
    dBA = nn_dist(B, A)

    # Chamfer（对称）：这里给两种口味
    dAB2, dBA2 = dAB**2, dBA**2
    chamfer_L2 = np.sqrt(np.mean(dAB2) + np.mean(dBA2))       # 常用展示：平方平均再开根（单位：米）
    chamfer_L2_sq = np.mean(dAB2) + np.mean(dBA2)             # 不开根（便于和论文对比）

    def stats(name, d):
        print(f"[{name}] mean={d.mean():.4f} m | median={np.median(d):.4f} m | p95={np.percentile(d,95):.4f} m | max={d.max():.4f} m | rmse={np.sqrt((d**2).mean()):.4f} m")

    print("\n=== 指标（单位：米）===")
    stats("A→B", dAB)
    stats("B→A", dBA)
    print(f"Chamfer(L2, 开根) = {chamfer_L2:.4f} m")
    print(f"Chamfer(L2, 平方和) = {chamfer_L2_sq:.6f} m^2")

    if src_sz and rec_sz:
        ratio = src_sz / max(rec_sz,1)
        print(f"\n[压缩比（按文件大小粗估）] {src_sz/1e6:.1f} MB → {rec_sz/1e6:.1f} MB  ≈ {ratio:.2f}x")

if __name__ == "__main__":
    main()
