import torch, numpy as np
from pathlib import Path
from SpareSPACE.train_ae3d import LogDensPatchDataset, AE3D

npz = Path(r"F:\PointCloudPython\data\raw_data\Lille1\Lille_0_vox0.10_B64_S48_log_q95_dens_sat.npz")
ckpt = Path(r"ae3d.ckpt")
idx = 0  # 你也可改成 10、50、100 看看

ds = LogDensPatchDataset(npz)
vol, i = ds[idx]
model = AE3D()
model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
model.eval()

with torch.no_grad():
    pred = model(vol.unsqueeze(0))
dens_hat = pred[0,0].numpy()
print(f"块 {idx}: min={dens_hat.min():.6f}, max={dens_hat.max():.6f}, mean={dens_hat.mean():.6f}, 非零比例={(dens_hat>1e-6).mean()*100:.2f}%")