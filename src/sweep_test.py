from load import DataLoader
import os, cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np

# from sweep import warp
from sweep.warp_soanr_image import warp, warp_matrix

# 使用示例
if __name__ == "__main__":
    # 替换为你的数据目录路径
    output_dir = "/home/clp/catkin_ws/src/sonar_cam_stereo/src/sweep/result_60"
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.dirname(os.path.abspath(__file__)) + "/data"
    
    # 创建数据加载器
    loader = DataLoader(data_path)
    
    # 读取第一帧数据
    rgb, depth, pose, sonar, sonar_rect = loader.load_frame(0)
    
    
    # 定义变换参数
    # depth = 3  # 深度值
    phi = 0.5 * 67.38 * np.pi/180  # 垂直视角
    pitch = 60 * np.pi/180  # 俯仰角（示例值）
    
    for i, depth in enumerate(np.arange(0, 10, 0.5)):
        print(f"处理深度值: {depth}")
        
        # 执行变换
        transformed_image = warp_matrix(sonar_rect, depth, phi, pitch)
        
        # 构建输出文件名
        output_filename = os.path.join(output_dir, f"depth_{depth:03f}.jpg")
        
        # 保存结果
        cv2.imwrite(output_filename, transformed_image)
        
        print(f"已保存: {output_filename}")
    
    # print("所有深度变换完成!")
    
    
    # cv2.imshow("sonar", np.flip(sonar, 0))
    # cv2.imshow("sonar_rect", np.flip(sonar_rect, 0))
    # cv2.waitKey(0)