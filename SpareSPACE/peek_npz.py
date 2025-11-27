import sys, numpy as np
from pathlib import Path
p = Path(sys.argv[1])
z = np.load(p, allow_pickle=False)
keys = [k for k in z.keys() if k.startswith("P")]
print("[INFO] 样本块数：", len(keys))
print("[INFO] VOX, BLOCK, STRIDE:", z["VOX"][0], z["BLOCK"][0], z["STRIDE"][0])
print("[INFO] QREF, DENS_QUANT:", z["QREF"][0], z["DENS_QUANT"][0])
print("[INFO] 第一块形状：", z[keys[0]].shape, "| 列含义=i,j,k,dens_q,sat")
print("[INFO] 第一块 Kref：", z["KREFS"][0])
# 取前5行看看
print(z[keys[0]][:5])
