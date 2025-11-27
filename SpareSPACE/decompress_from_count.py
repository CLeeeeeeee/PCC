# 用法示例：
#   python decompress_from_count.py F:\PointCloudPython\data\raw_data\Lille1\Lille_0_voxel0.20_centroid_with_count.ply --voxel 0.20 --origin_mode min --out F:\PointCloudPython\data\raw_data\Lille1\Lille_0_recon.ply --preview
# 依赖：pip install numpy plyfile open3d

import argparse, re, numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
import open3d as o3d

def read_rep_count(path: Path):
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    pts = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    cnt = np.asarray(v['count'], dtype=np.uint32)
    I   = np.asarray(v['intensity'], dtype=np.float32) if 'intensity' in v.dtype.names else None
    origin, voxel = None, None
    # 尝试从注释解析
    for c in getattr(ply, "comments", []):
        if c.startswith("origin:"):
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", c)
            if len(nums)>=3:
                origin = np.array(nums[:3], dtype=np.float64)
        if c.startswith("voxel_size_m:"):
            vs = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", c)
            if vs:
                voxel = float(vs[0])
    return pts, cnt, I, origin, voxel

# 简单 Halton 序列（蓝噪声风格均匀）
def _halton(n, base):
    f, r = 1.0, 0.0
    while n > 0:
        f /= base
        r += f * (n % base)
        n //=  base
    return r
def halton_3d(N, start=1):
    idx = np.arange(start, start+N, dtype=np.int64)
    h2 = np.vectorize(_halton)(idx, 2)
    h3 = np.vectorize(_halton)(idx, 3)
    h5 = np.vectorize(_halton)(idx, 5)
    return np.stack([h2, h3, h5], axis=1)

def decompress_uniform(rep_pts, counts, origin, voxel_size, intensity=None, jitter_scale=0.2, cap_per_voxel=None):
    # 体素整数坐标
    vij = np.floor((rep_pts - origin) / voxel_size).astype(np.int64)
    base = origin + vij * voxel_size
    out_pts = []
    out_I   = [] if intensity is not None else None

    # 为每体素生成不同的起始索引种子，避免同分布
    seeds = (vij[:,0]*73856093 ^ vij[:,1]*19349663 ^ vij[:,2]*83492791) & 0xFFFFFFFF

    for m, k in enumerate(counts):
        k = int(k)
        if cap_per_voxel is not None:
            k = min(k, cap_per_voxel)  # 可选：防止单体素爆点
        if k <= 0:
            continue
        if k == 1:
            out_pts.append(rep_pts[m:m+1])
            if out_I is not None: out_I.append(np.full((1,), intensity[m], dtype=np.float32))
            continue
        h = halton_3d(k, start=int(seeds[m]%100000 + 1))
        rng = np.random.default_rng(seeds[m])
        jitter = (rng.random((k,3)) - 0.5) * jitter_scale   # 轻微扰动
        u = np.clip(h + jitter, 0, 1)
        pts = base[m] + u * voxel_size
        out_pts.append(pts)
        if out_I is not None:
            out_I.append(np.full((k,), intensity[m], dtype=np.float32))
    P = np.vstack(out_pts) if out_pts else np.zeros((0,3), dtype=np.float32)
    I = np.concatenate(out_I) if out_I is not None and out_pts else None
    return P, I

def write_ply(path: Path, pts, intensity=None):
    N = pts.shape[0]
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    if intensity is not None:
        dtype.append(('intensity', np.float32))
    data = np.empty(N, dtype=dtype)
    data['x'], data['y'], data['z'] = pts[:,0], pts[:,1], pts[:,2]
    if intensity is not None:
        data['intensity'] = intensity
    PlyData([PlyElement.describe(data,'vertex')], text=True).write(str(path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep_ply", type=str)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--cap_per_voxel", type=int, default=None)
    ap.add_argument("--voxel", type=float, default=None, help="若PLY注释缺失，请手动指定体素边长(米)")
    ap.add_argument("--origin", type=float, nargs=3, default=None, help="若PLY注释缺失，手动指定origin: ox oy oz")
    ap.add_argument("--origin_mode", choices=["min","zero"], default="min",
                    help="若无注释且未手动给origin时的兜底：min=用代表点最小值；zero=用(0,0,0)")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    rep, cnt, I, origin, voxel = read_rep_count(Path(args.rep_ply))

    # 兜底：优先用命令行 --voxel / --origin
    if voxel is None:
        voxel = args.voxel
    if origin is None:
        origin = np.array(args.origin, dtype=np.float64) if args.origin is not None else None
    if voxel is None:
        raise ValueError("缺少 voxel_size：可在命令行加 --voxel 0.20")
    if origin is None:
        origin = rep.min(axis=0).astype(np.float64) if args.origin_mode=="min" else np.zeros(3, dtype=np.float64)

    recon, Irec = decompress_uniform(rep, cnt, origin, voxel, intensity=I,
                                     jitter_scale=0.2, cap_per_voxel=args.cap_per_voxel)

    out = Path(args.out) if args.out else Path(args.rep_ply).with_name(Path(args.rep_ply).stem + "_recon.ply")
    write_ply(out, recon, Irec)
    print(f"[DONE] 重建写出：{out} | 重建点数={recon.shape[0]:,}")

    if args.preview:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(recon)
        if Irec is not None:
            I0 = Irec.copy()
            lo, hi = np.percentile(I0, [1,99])
            I0 = np.clip((I0 - lo)/max(hi-lo,1e-6), 0, 1)
            pcd.colors = o3d.utility.Vector3dVector(np.stack([I0,I0,I0],1))
        else:
            pcd.paint_uniform_color([0.2,0.4,0.8])
        pcd = pcd.voxel_down_sample(max(voxel, 0.10))
        o3d.visualization.draw_geometries([pcd], window_name="Reconstructed", width=1280, height=800)

if __name__ == "__main__":
    main()
