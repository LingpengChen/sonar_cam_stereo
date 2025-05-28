from load import DataLoader
import os, cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np

# from sweep import warp
from sweep.warp_soanr_image import warp, warp_matrix, warp_differentiable

import torch
import cv2
import numpy as np
import os
from torchvision import transforms
import matplotlib.pyplot as plt

def load_image(image_path):
    """加载图像并转换为PyTorch张量"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img)
    return img_tensor.unsqueeze(0)  # 添加批次维度 [1, C, H, W]

# 使用示例
if __name__ == "__main__":
    # 替换为你的数据目录路径
    output_dir = "/home/clp/catkin_ws/src/sonar_cam_stereo/src/sweep/result_test"
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.dirname(os.path.abspath(__file__)) + "/data"
    
    # 创建数据加载器
    loader = DataLoader(data_path)
    
    # 读取第一帧数据
    rgb, depth, pose, sonar, sonar_rect = loader.load_frame(0)
    
    sonar_rect_tensor = transforms.ToTensor()(sonar_rect).unsqueeze(0) 
    # 定义变换参数
    # depth = 3  # 深度值
    phi = 0.5 * 67.38 * np.pi/180  # 垂直视角
    pitch = 60 * np.pi/180  # 俯仰角（示例值）
    
    for i, depth in enumerate(np.arange(1.5, 10, 0.5)):
        print(f"处理深度值: {depth}")
        
        # 执行变换
        # transformed_image = warp_differentiable(sonar_rect, depth, phi, pitch)
        transformed_image = warp_differentiable(sonar_rect_tensor, depth, pitch)
        
        warped_np = transformed_image[0,0].detach().numpy() 
        # warped_np = warped_np.astype(np.uint8)
        # 构建输出文件名
        output_filename = os.path.join(output_dir, f"depth_{depth:03f}.png")
        
        plt.imshow(warped_np)
        plt.title("warped")
        plt.colorbar()
        plt.axis('on')
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存: {output_filename}")
