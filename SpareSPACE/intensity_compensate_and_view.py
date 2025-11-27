# 文件名：intensity_compensate_and_view.py
# 依赖：pip install open3d plyfile numpy

import numpy as np
from plyfile import PlyData
import open3d as o3d
import sys
from pathlib import Path

# ----------- 1) 输入文件 ------------
ply_path = Path("sample_xyz_i_rgb.ply" if len(sys.argv) < 2 else sys.argv[1])
if not ply_path.exists():
    raise FileNotFoundError(f"找不到 PLY 文件：{ply_path}")

# ----------- 2) 读取 XYZ / RGB / I ------------
ply = PlyData.read(str(ply_path))
v = ply["vertex"].data

xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)              # (N,3)
rgb = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.float32)     # 0..255
rgb = rgb / 255.0
Iraw = np.array(v["intensity"], dtype=np.float32)

# ----------- 3) 距离补偿 I1 = Iraw * r^2 ------------
r = np.linalg.norm(xyz, axis=1) + 1e-6
I1 = Iraw * (r ** 2)

# ----------- 4) 估法线 & 入射角补偿 I2 ------------
pcd_tmp = o3d.geometry.PointCloud()
pcd_tmp.points = o3d.utility.Vector3dVector(xyz)
pcd_tmp.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30)
)
normals = np.asarray(pcd_tmp.normals)  # (N,3)

# beam方向：传感器坐标系中，射线近似=从原点指向点的反向单位向量
beam_dir = -xyz / (np.linalg.norm(xyz, axis=1, keepdims=True) + 1e-6)
cos_theta = np.clip(np.sum(beam_dir * normals, axis=1), 1e-3, 1.0)

alpha = 1.5  # 可调参数，1.0~2.0 常见，后续可网格搜索
I2 = I1 * (cos_theta ** (-alpha))

# ----------- 5) 帧内归一化（分位截断）→ I* ------------
lo, hi = np.percentile(I2, [1, 99])
I_star = np.clip((I2 - lo) / max(hi - lo, 1e-6), 0, 1)

# ----------- 6) 构建两个点云视图：RGB & Intensity灰度 ------------
pcd_RGB = o3d.geometry.PointCloud()
pcd_RGB.points = o3d.utility.Vector3dVector(xyz)
pcd_RGB.colors = o3d.utility.Vector3dVector(rgb)

gray = np.stack([I_star, I_star, I_star], axis=1)
pcd_I = o3d.geometry.PointCloud()
pcd_I.points = o3d.utility.Vector3dVector(xyz)
pcd_I.colors = o3d.utility.Vector3dVector(gray)

# ----------- 7) 可选：保存校正后的 intensity 为伪彩色 PLY ------------
# （如果你希望导出结果以便别的工具查看，把下面两行取消注释）
# o3d.io.write_point_cloud("out_intensity_gray.ply", pcd_I)
# o3d.io.write_point_cloud("out_rgb.ply", pcd_RGB)

# ----------- 8) 可视化（指定窗口大小，避免警告） ------------
o3d.visualization.draw_geometries([pcd_RGB], window_name="RGB view", width=1024, height=768)
o3d.visualization.draw_geometries([pcd_I],   window_name="Intensity (r^2 + angle) view", width=1024, height=768)

print("完成：RGB视图 和 强度灰度视图 已显示。可调参数 alpha =", alpha)
