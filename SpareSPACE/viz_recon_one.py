# viz_recon_one.py
# 用法：
#   python viz_recon_one.py --npz <npz路径> --ckpt ae3d.ckpt --idx 0 --out recon_block0.ply
# 依赖：pip install numpy torch plyfile

import argparse, numpy as np, torch
from pathlib import Path
from plyfile import PlyData, PlyElement

from SpareSPACE.train_ae3d import LogDensPatchDataset, AE3D

def write_ply_xyz(path, pts):
    N = pts.shape[0]
    data = np.empty(N, dtype=[('x', 'f4'), ('y','f4'), ('z','f4')])
    data['x'], data['y'], data['z'] = pts[:,0], pts[:,1], pts[:,2]
    PlyData([PlyElement.describe(data, 'vertex')], text=True).write(str(path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--idx", type=int, default=0, help="要可视化的块索引")
    ap.add_argument("--th", type=float, default=0.05, help="密度门限（仅用于决定是否采样）")
    ap.add_argument("--jitter", type=float, default=0.3, help="体素内抖动比例(0..1)")
    ap.add_argument("--out", type=str, default="recon_block.ply")
    args = ap.parse_args()

    ds = LogDensPatchDataset(Path(args.npz))
    B = ds.block
    # 取该样本
    vol, idx = ds[args.idx]     # vol: [2,64,64,64]
    vol = vol.unsqueeze(0)      # [1,2,D,H,W]
    kref = ds.krefs[idx]        # 该块的 Kref
    log1 = np.log1p(float(kref))

    # 模型
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = AE3D(); model.load_state_dict(ckpt["model"]); model.eval()

    with torch.no_grad():
        pred = model(vol)       # [1,2,64,64,64]
        dens_hat = pred[0,0].numpy()  # [64,64,64] in [0,1]
        # 反变换到期望计数
        c_hat = np.exp(dens_hat * log1) - 1.0
        # 小门限：太小的密度忽略
        mask = dens_hat >= args.th
        ijk = np.stack(np.nonzero(mask), axis=1)  # (M,3)
        if ijk.size == 0:
            print("该块阈值后为空，换一个 idx 试试"); return
        # 每个体素撒 round(c_hat) 个点（上限做个cap避免超多）
        counts = np.clip(np.round(c_hat[mask]).astype(int), 1, 8)
        # 体素中心坐标（以块局部坐标系表示）
        centers = ijk + 0.5
        # 生成点
        pts = []
        rng = np.random.default_rng(0)
        for p, k in zip(centers, counts):
            # 抖动范围：jitter * 体素边长；这里体素边长先设为1个单位，后续你可乘以真实 VOX
            if k == 1:
                pts.append(p)
            else:
                jitter = (rng.random((k,3)) - 0.5) * args.jitter  # [-j/2, j/2]
                pts.append(p + jitter)
        P = np.vstack(pts).astype(np.float32)
        write_ply_xyz(Path(args.out), P)
        print(f"[DONE] 写出：{args.out}  点数={P.shape[0]}  (注意：此处坐标是块内单位坐标，非米)")
        print("提示：可用 Open3D 打开看看结构；若要换回米制坐标，在后续全局拼接时乘以 VOX 并加上块的起点。")

if __name__ == "__main__":
    main()
