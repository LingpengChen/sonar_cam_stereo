import numpy as np
import cv2, os, math
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

# 相机内参
fx = 360
fy = 360
cx = 360
cy = 240

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
# 计算相机内参逆矩阵
K_inv = np.linalg.inv(K)

# # 获取原图像尺寸
# src_height, src_width = sonar_rect.shape[:2]
# 创建目标图像
height = 480
width = 720

def warp(sonar_rect, depth, phi, alpha):
    """
    将原图像通过给定的投影关系变换到目标图像
    
    参数:
    sonar_rect: 原始图像 (numpy数组)
    depth: 深度值
    phi: 垂直视角（弧度）
    alpha: 俯仰角（弧度）
    
    返回:
    transformed_image: 变换后的图像
    """
    # 相机内参
    fx = 360
    fy = 360
    cx = 360
    cy = 240
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # 获取原图像尺寸
    src_height, src_width = sonar_rect.shape[:2]
    
    # 创建目标图像
    height = 480
    width = 720
    if len(sonar_rect.shape) == 3:  # 彩色图像
        result = np.zeros((height, width, 3), dtype=np.uint8)
    else:  # 灰度图像
        result = np.zeros((height, width), dtype=np.uint8)
    
    # 计算相机内参逆矩阵
    K_inv = np.linalg.inv(K)
    
    def transform_coordinates(u, v):
        # 目标图像中的点 p_c
        p_c = np.array([u, v, 1])
        
        # 通过 K 的逆矩阵得到归一化坐标
        p_normalized = K_inv @ p_c  # [x/y, (x-depth)tan/y, 1]
        
        # 通过逆向投影过程计算3D点坐标
        a = p_normalized[0] 
        b = p_normalized[1] 
        
        theta = np.arctan(a)
        d = depth*np.tan(alpha)/(b+np.tan(alpha))
        
        theta_prime = 8 * (np.rad2deg(theta)+45)
        d_prime = 100 * d
        
        return theta, d, theta_prime, d_prime
  
    # 对目标图像的每个像素进行处理
    for v in range(height):
        for u in range(width):
            _, _, u_prime, v_prime = transform_coordinates(u, v)
            
            # 检查坐标是否在原图像范围内
            if 0 <= u_prime < src_width - 1 and 0 <= v_prime < src_height - 1:
                # 双线性插值
                
                # 获取周围四个像素的整数坐标
                u_floor = int(np.floor(u_prime))
                v_floor = int(np.floor(v_prime))
                u_ceil = u_floor + 1
                v_ceil = v_floor + 1
                
                # 确保索引不超出图像边界
                if u_ceil >= src_width:
                    u_ceil = src_width - 1
                if v_ceil >= src_height:
                    v_ceil = src_height - 1
                
                # 计算插值权重
                u_weight = u_prime - u_floor
                v_weight = v_prime - v_floor
                
                # 执行双线性插值
                if len(sonar_rect.shape) == 3:  # 彩色图像
                    # 提取四个角点的像素值
                    p1 = sonar_rect[v_floor, u_floor].astype(float)
                    p2 = sonar_rect[v_floor, u_ceil].astype(float)
                    p3 = sonar_rect[v_ceil, u_floor].astype(float)
                    p4 = sonar_rect[v_ceil, u_ceil].astype(float)
                    
                    # 双线性插值计算
                    pixel_value = (1 - u_weight) * (1 - v_weight) * p1 + \
                                  u_weight * (1 - v_weight) * p2 + \
                                  (1 - u_weight) * v_weight * p3 + \
                                  u_weight * v_weight * p4
                    
                    result[v, u] = pixel_value.astype(np.uint8)
                else:  # 灰度图像
                    # 提取四个角点的像素值
                    p1 = float(sonar_rect[v_floor, u_floor])
                    p2 = float(sonar_rect[v_floor, u_ceil])
                    p3 = float(sonar_rect[v_ceil, u_floor])
                    p4 = float(sonar_rect[v_ceil, u_ceil])
                    
                    # 双线性插值计算
                    pixel_value = (1 - u_weight) * (1 - v_weight) * p1 + \
                                  u_weight * (1 - v_weight) * p2 + \
                                  (1 - u_weight) * v_weight * p3 + \
                                  u_weight * v_weight * p4
                    
                    result[v, u] = int(pixel_value)
    
    return result


def warp_matrix(sonar_rect, depth, alpha):
    """
    使用矩阵操作进行图像变换
    
    参数:
    sonar_rect: 原始图像 (numpy数组)
    depth: 深度值
    phi: 垂直视角（弧度）
    alpha: 俯仰角（弧度）
    
    返回:
    transformed_image: 变换后的图像
    """

    # 创建网格坐标
    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # 将坐标转换成齐次坐标
    ones = np.ones_like(u_coords)
    pixel_coords = np.stack([u_coords, v_coords, ones], axis=-1)  # 形状为 (height, width, 3)
    
    # 重塑为 (height*width, 3) 的矩阵，方便矩阵乘法
    pixel_coords_reshaped = pixel_coords.reshape(-1, 3)
    
    # 使用相机内参的逆矩阵获取归一化坐标
    normalized_coords = (K_inv @ pixel_coords_reshaped.T).T  # 形状为 (height*width, 3)
    
    # 提取归一化坐标的分量
    a = normalized_coords[:, 0]
    b = normalized_coords[:, 1]
    
    # 计算 theta 和 d
    theta = np.arctan(a)
    d = depth * np.tan(alpha) / (b + np.tan(alpha))
    
    # 计算原图像中的坐标
    theta_prime = 8 * (np.rad2deg(theta) + 45)
    d_prime = 100 * d
    
    # 重塑为原始网格形状并填充映射矩阵
    map_x = theta_prime.reshape(height, width).astype(np.float32)
    map_y = d_prime.reshape(height, width).astype(np.float32)
    
    # 使用OpenCV的remap函数进行重映射
    result = cv2.remap(sonar_rect, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return result


def warp_differentiable(sonar_rect, depth, alpha):
    """
    使用PyTorch实现可微分的图像变换函数
    
    参数:
    sonar_rect: PyTorch张量，形状为[batch, channels, height, width]
    depth: 深度值（标量或张量）
    alpha: 俯仰角（弧度，标量或张量）
    
    返回:
    transformed_image: 变换后的图像，PyTorch张量
    """
    # 确保输入是PyTorch张量
    if not isinstance(sonar_rect, torch.Tensor):
        sonar_rect = torch.tensor(sonar_rect, dtype=torch.float32)
    
    # 添加批次和通道维度(如果需要)
    if len(sonar_rect.shape) == 2:
        sonar_rect = sonar_rect.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
    elif len(sonar_rect.shape) == 3:
        sonar_rect = sonar_rect.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    
    # 获取图像尺寸
    batch, channels, src_height, src_width = sonar_rect.shape
    
    # 目标图像大小
    height = 480
    width = 720
    
    # 相机内参
    fx = 360
    fy = 360
    cx = 360
    cy = 240
    
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=sonar_rect.device)
    
    # 计算相机内参逆矩阵
    K_inv = torch.inverse(K)
    
    # 创建目标图像中的坐标网格
    y_coords = torch.arange(0, height, device=sonar_rect.device).float()
    x_coords = torch.arange(0, width, device=sonar_rect.device).float()
    
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # 将坐标转换为齐次坐标
    ones = torch.ones_like(x_coords)
    pixel_coords = torch.stack([x_coords, y_coords, ones], dim=-1)  # [H,W,3]
    
    # 应用内参逆矩阵获取归一化坐标
    # 对于每个像素应用矩阵乘法
    normalized_coords = torch.matmul(pixel_coords.view(-1, 3), K_inv.T)  # [H*W,3]
    
    # 提取归一化坐标分量
    a = normalized_coords[:, 0]
    b = normalized_coords[:, 1]
    
    # 计算theta和d
    theta = torch.atan(a)
    
    # 确保alpha是张量
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float32, device=sonar_rect.device)
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth, dtype=torch.float32, device=sonar_rect.device)
        
    d = depth * torch.tan(alpha) / (b + torch.tan(alpha))
    
    # 计算原图像中的坐标
    theta_prime = 8 * (theta * 180 / math.pi + 45)
    d_prime = 100 * d
    
    # 重塑为原始网格形状
    map_x = theta_prime.view(height, width)
    map_y = d_prime.view(height, width)
    
    # 将像素坐标从[0, width/height]范围转换为[-1, 1]范围，适用于grid_sample
    map_x = 2.0 * map_x / (src_width - 1) - 1.0
    map_y = 2.0 * map_y / (src_height - 1) - 1.0
    
    # 合并成采样网格
    grid = torch.stack([map_x, map_y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    
    # 使用grid_sample进行可微分的重采样
    result = F.grid_sample(sonar_rect, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    

    return result