import open3d as o3d

pcd = o3d.io.read_point_cloud("sample_xyz_i_rgb.ply")

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="PLY view", width=960, height=720, visible=True)
vis.add_geometry(pcd)
vis.run()              # 进入交互
vis.destroy_window()
