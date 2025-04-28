import cv2
import numpy as np
import os
import glob
import math
from typing import Tuple

intrisic = np.array([[554.25624847,   0.,         320.,        ],
                    [  0.,         554.25624847, 240.        ],
                    [  0.,           0.,           1.        ]])

def load_depth_from_npy(filepath):
    """
    从.npy文件加载深度图像数据
    
    参数:
        filepath: .npy文件的路径
    
    返回:
        加载的深度图像数据
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    # 加载数据
    depth_data = np.load(filepath)['depth']
    print(f"已从 {filepath} 加载深度数据，形状: {depth_data.shape}")
    
    return depth_data


def visualize_depth(depth_data):
    """
    将深度数据可视化为彩色图像，nan值显示为黑色
    
    参数:
        depth_data: 深度图像数据
    
    返回:
        彩色深度图
    """
    # 创建深度数据的副本
    depth = depth_data.copy()
    
    # 获取nan的掩码
    nan_mask = np.isnan(depth)
    
    # 临时将nan替换为0以便进行归一化
    depth[nan_mask] = 0
    
    # 归一化到0-1范围
    normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    
    # 转换为0-255的uint8类型
    normalized = (normalized * 255).astype(np.uint8)
    
    # 应用色彩映射
    depth_colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    # 将nan对应的像素设置为黑色 [0,0,0]
    depth_colormap[nan_mask] = [0,0,0]
    
    return depth_colormap


def generate_sonar_view(depth_data, fx=360, cx=360, max_depth=10.0, gamma=0.7):
    """
    从透视深度图生成声纳视图，保持原始图像的宽度，应用余弦校正
    
    参数:
        depth_data: 深度图像数据 (H, W) 的numpy数组，值代表沿视线的距离
        fx: 相机内参中的水平焦距
        cx: 相机内参中的水平主点坐标，如果为None则使用图像中心
        max_depth: 最大深度值截断，默认为10米
        gamma: 伽马校正系数
        
    返回:
        声纳视图图像
    """
    height, half_width = 640, 453

    
    # 创建声纳图（初始为全0），保持原始宽度
    sonar_image = np.zeros((height, 2*half_width), dtype=np.float32)

    v_indices, u_indices = np.where((depth_data > 0) & (depth_data <= max_depth) & ~np.isnan(depth_data))
    depths = depth_data[v_indices, u_indices]
    
    # 计算归一化的方向向量的x分量 tan(alpha )
    dx = (u_indices - cx) / fx
    
    # 计算视线方向与Z轴夹角的余弦值
    cos_alpha = 1.0 / np.sqrt(dx**2 + 1.0)
    sin_alpha = dx / np.sqrt(dx**2 + 1.0)
    
    
    u_sonar = half_width + (height * depths * sin_alpha / max_depth).astype(int)
    v_sonar = (height  * (1 -  (depths * cos_alpha) / max_depth) ).astype(int)
    
    # 在声纳图像中标记点（累积密度）
    # 使用深度值作为权重，更远的点权重更小
    # weights = 1.0 / (1.0 + depths)  # 简单的距离衰减公式
    weights = depths  # 简单的距离衰减公式
    
    for u, v, w in zip(u_sonar, v_sonar, weights):
        sonar_image[v][u] += w**2
        
    # 图像归一化
    sonar_image = sonar_image / np.max(sonar_image)
    
    # 应用高斯模糊使点更明显
    # sonar_image = cv2.GaussianBlur(sonar_image, (5, 5), 0)
    
    # # 应用伽马校正增强对比度
    # sonar_image = np.power(sonar_image, gamma)
    
    sonar_color = cv2.applyColorMap(
            (255 * sonar_image).astype(np.uint8), 
            cv2.COLORMAP_HOT
        )
    
    return sonar_image, sonar_color

def get_rectangular_image(depth_data:np.ndarray,
                          n_ranges: int = 1000, 
                          n_azimuths: int = 720,
                          max_range: int = 10):
    '''
        n_ranges: the number of range samples, also the heigh of rectanglar sonar image
        n_azimuths: the number of azimuth samples, also the width of rectanglar sonar image
        max_range: sonar detection range (max depth of depth image)
    '''
    # width = depth_data.shape[1]
    sonar_image = np.zeros((n_ranges, n_azimuths), dtype=np.float32)
    
    v_indices, u_indices = np.where(~np.isnan(depth_data))
    depths = depth_data[v_indices, u_indices]
    theta = u_indices
    for theta, d in zip(theta, depths):
        range = int( n_ranges * d / max_range)
        sonar_image[range][theta] += range
    sonar_image = sonar_image / np.max(sonar_image)
    # sonar_image = np.power(sonar_image, 0.7)
    
    sonar_color = cv2.applyColorMap(
            (255 * sonar_image).astype(np.uint8), 
            cv2.COLORMAP_HOT
        )
    
    return sonar_image, sonar_color


def rect_to_sonar_map(rect_image: np.ndarray,
                      n_ranges: int = 1000, 
                      n_azimuths: int = 720,
                      azimuth_bounds: Tuple[float, float] = (-0.785, 0.785)) -> np.ndarray:
    """
    将矩形声纳图像映射为扇形图像
    
    Args:
        n_ranges: 距离采样点数量
        n_azimuths: 方位角采样点数量  
        azimuth_bounds: 方位角范围(min_angle, max_angle)
        rect_image: 输入的矩形图像
        
    Returns:
        扇形声纳图像
    """
    # 计算目标图像尺寸
    minus_width = math.floor(n_ranges * math.sin(azimuth_bounds[0]))
    plus_width = math.ceil(n_ranges * math.sin(azimuth_bounds[1]))
    width = plus_width - minus_width
    

    # 创建映射矩阵
    map_x = np.zeros((n_ranges, width), dtype=np.float32)
    map_y = np.zeros((n_ranges, width), dtype=np.float32)
    
    # 计算方位角步长
    db = (azimuth_bounds[1] - azimuth_bounds[0]) / n_azimuths
    origin_x = abs(minus_width)
    
    # 计算映射关系
    for x in range(width):
        for y in range(n_ranges):
            dx = x - origin_x
            dy = y
            
            range_val = math.sqrt(dx * dx + dy * dy)
            azimuth = math.atan2(dx, dy)
            
            map_x[y, x] = (azimuth - azimuth_bounds[0]) / db
            map_y[y, x] = range_val
    
    # 执行重映射
    sonar_image = cv2.remap(rect_image, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return sonar_image

if __name__ == "__main__":

    # 获取目录中所有的.npy文件
    depth_data_dir = "/home/clp/catkin_ws/src/sonar_cam_stereo/src/data/depth"
    npy_files = glob.glob(os.path.join(depth_data_dir, "*.npz"))
    npy_files = sorted(npy_files)

    for current_file in npy_files:
        filename = os.path.basename(current_file)
        depth_data = load_depth_from_npy(current_file)
        depth_colormap = visualize_depth(depth_data)
        cv2.imshow("Depth Image", depth_colormap)
        
        # sonar_image, sonar_color = generate_sonar_view(depth_data)
        sonar_image_rect, sonar_color_rect = get_rectangular_image(depth_data)
        
        # cv2.imshow will set upper left corner as (0,0), 
        # for better visualization we set bottom left as (0,0) by flipping
        cv2.imshow("sonar_color_rect",  cv2.flip(sonar_color_rect, 0))
      
        sonar_color = rect_to_sonar_map(sonar_color_rect)

        cv2.imshow("sonar_color", cv2.flip(sonar_color, 0))
        cv2.waitKey(0)
        break