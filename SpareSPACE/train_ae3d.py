# train_ae3d.py
# 用法：
#   python train_ae3d.py F:\PointCloudPython\data\raw_data\Lille1\Lille_0_vox0.10_B64_S48_log_q95_dens_sat.npz --epochs 10 --bs 4
# 依赖：pip install torch numpy

import argparse, math, random, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ===== Dataset：把稀疏块落到 dense 64x64x64x2 =====
class LogDensPatchDataset(Dataset):
    def __init__(self, npz_path):
        self.z = np.load(npz_path, allow_pickle=False)
        self.keys = sorted([k for k in self.z.keys() if k.startswith("P")], key=lambda x:int(x[1:]))
        self.block = int(self.z["BLOCK"][0])
        fmt = int(self.z["FORMAT"][0])  # 3: i,j,k,dens_q,sat
        assert fmt == 3, f"FORMAT={fmt} 不匹配（期望3）"
        # 每块一个 Kref，后续可用在可视化反变换，这里训练阶段不用
        self.krefs = self.z["KREFS"].astype(np.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        arr = self.z[self.keys[idx]]   # (n,5): i,j,k,dens_q,sat
        B = self.block
        # dense [C=2, D, H, W]
        vol = np.zeros((2, B, B, B), dtype=np.float32)
        i, j, k = arr[:,0].astype(np.int64), arr[:,1].astype(np.int64), arr[:,2].astype(np.int64)
        dens = (arr[:,3].astype(np.float32) / 255.0)          # [0,1]
        sat  = arr[:,4].astype(np.float32)                    # {0,1}
        vol[0, i, j, k] = np.clip(dens, 0.0, 1.0)             # dens
        vol[1, i, j, k] = sat                                 # sat
        return torch.from_numpy(vol), idx  # 返回 idx 方便后续可视化时取 Kref

# ====== 简单 3D 卷积自编码器 ======
class AE3D(nn.Module):
    def __init__(self, in_ch=2, base=16, bottleneck_ch=8):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 64 -> 32
            nn.Conv3d(base, base*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 32 -> 16
            nn.Conv3d(base*2, base*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 16 -> 8
            nn.Conv3d(base*4, bottleneck_ch, 1)   # 8x8x8 x bottleneck_ch
        )
        # Decoder（镜像上采样）
        self.dec = nn.Sequential(
            nn.Conv3d(bottleneck_ch, base*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),  # 8->16
            nn.Conv3d(base*4, base*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),  # 16->32
            nn.Conv3d(base*2, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),  # 32->64
            nn.Conv3d(base, 2, 1)   # 输出2通道：dens_hat(未激活), sat_hat(未激活)
        )

    def forward(self, x):
        z = self.enc(x)
        y = self.dec(z)
        # 输出两通道：dens 用 sigmoid 到 [0,1]，sat 也 sigmoid（做概率）
        dens_hat = torch.sigmoid(y[:,0:1])
        sat_hat  = torch.sigmoid(y[:,1:2])
        return torch.cat([dens_hat, sat_hat], dim=1)

# ====== 训练入口 ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--sat_w", type=float, default=0.2, help="sat 通道的损失权重")
    ap.add_argument("--save", type=str, default="ae3d.ckpt")
    args = ap.parse_args()

    ds = LogDensPatchDataset(Path(args.npz))
    n = len(ds)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    tr, va = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    dl_tr = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=True)
    dl_va = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE3D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # dens: Huber；sat: BCE
    huber = nn.SmoothL1Loss()
    bce   = nn.BCELoss()

    best = float("inf")
    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for vol, _idx in dl_tr:
            vol = vol.to(device)  # [B,2,64,64,64]
            pred = model(vol)
            loss_d = huber(pred[:,0:1], vol[:,0:1])
            loss_s = bce(pred[:,1:2],  vol[:,1:2])
            loss = loss_d + args.sat_w * loss_s
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * vol.size(0)
        tr_loss /= n_train

        # 验证
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for vol, _idx in dl_va:
                vol = vol.to(device)
                pred = model(vol)
                loss_d = huber(pred[:,0:1], vol[:,0:1])
                loss_s = bce(pred[:,1:2],  vol[:,1:2])
                loss = loss_d + args.sat_w * loss_s
                va_loss += loss.item() * vol.size(0)
        va_loss /= n_val

        print(f"[E{ep:02d}] train={tr_loss:.5f}  val={va_loss:.5f}")

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict()}, args.save)
            print(f"  ↳ saved to {args.save} (best so far)")

if __name__ == "__main__":
    main()
