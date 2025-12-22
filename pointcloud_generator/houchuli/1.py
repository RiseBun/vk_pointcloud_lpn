import open3d as o3d
import glob
import os
import numpy as np

# ================= 配置区域 =================
INPUT_DIR = "/home/li/pcd_final_gray"
OUTPUT_FILE = "/home/li/final_global_map.pcd"

# [核心] 如果想更精细，把这个改小 (比如 0.03)
VOXEL_SIZE = 0.03  

# [核心] 统计去噪
ENABLE_SOR = True
SOR_NEIGHBORS = 50
SOR_STD_RATIO = 1.0

# [新增] 半径去噪 (专门杀飞点)
ENABLE_ROR = True
ROR_POINTS = 6
ROR_RADIUS = 0.15
# ===========================================

def main():
    pattern = os.path.join(INPUT_DIR, "*_submap_*.pcd")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"[Error] 找不到文件！: {INPUT_DIR}")
        return

    print(f"[Info] 发现 {len(files)} 个子地图，开始合并...")
    combined_cloud = o3d.geometry.PointCloud()

    for i, f in enumerate(files):
        try:
            pcd = o3d.io.read_point_cloud(f)
            if not pcd.is_empty():
                combined_cloud += pcd
            if i % 10 == 0: print(f"\r[Loading] {i+1}/{len(files)}", end="")
        except: pass

    print(f"\n[Done] 原始点数: {len(combined_cloud.points)}")
    if combined_cloud.is_empty(): return

    # 1. 降采样
    print(f"[Process] 降采样 (Grid={VOXEL_SIZE})...")
    combined_cloud = combined_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)

    # 2. 统计滤波
    if ENABLE_SOR:
        print(f"[Process] 统计滤波 (SOR)...")
        cl, _ = combined_cloud.remove_statistical_outlier(nb_neighbors=SOR_NEIGHBORS, std_ratio=SOR_STD_RATIO)
        combined_cloud = cl

    # 3. 半径滤波 (新增)
    if ENABLE_ROR:
        print(f"[Process] 半径滤波 (ROR)...")
        cl, _ = combined_cloud.remove_radius_outlier(nb_points=ROR_POINTS, radius=ROR_RADIUS)
        combined_cloud = cl

    # 4. 保存
    o3d.io.write_point_cloud(OUTPUT_FILE, combined_cloud, write_ascii=False, compressed=True)
    print(f"[Success] 保存成功: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()