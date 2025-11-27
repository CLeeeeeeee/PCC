# 用法: python plot_sweep.py F:\PointCloudPython\data\raw_data\Lille1\sweep\results_voxel_sweep.csv
import sys, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path(sys.argv[1])
df = pd.read_csv(csv_path)
df = df.sort_values("voxel_m")

print(df[["voxel_m","point_compress_ratio","A2B_mean_m","B2A_mean_m","Chamfer_L2_m"]])

# 图1：误差 vs 体素
plt.figure()
plt.plot(df["voxel_m"], df["A2B_mean_m"], marker="o", label="A→B mean (m)")
plt.plot(df["voxel_m"], df["B2A_mean_m"], marker="o", label="B→A mean (m)")
plt.plot(df["voxel_m"], df["Chamfer_L2_m"], marker="o", label="Chamfer L2 (m)")
plt.xlabel("Voxel size (m)")
plt.ylabel("Error (m)")
plt.title("Error vs Voxel size")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()

# 图2：压缩比 vs 体素（点数压缩）
plt.figure()
plt.plot(df["voxel_m"], df["point_compress_ratio"], marker="o", label="Point compress ratio")
plt.xlabel("Voxel size (m)")
plt.ylabel("Compression ratio (×)")
plt.title("Point compression ratio vs Voxel size")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()
