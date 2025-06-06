import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_depth_3d(depth_data):
    """
    将深度数据可视化为3D表面图
    
    参数:
        depth_data: 深度图像数据
    """
    # 创建深度数据的副本
    depth = 32 / depth_data.copy()
    
    # 获取nan的掩码
    nan_mask = np.isnan(depth)
    
    # 将nan替换为0
    depth[nan_mask] = 0
    
    # 创建网格坐标
    rows, cols = depth.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D表面
    surf = ax.plot_surface(x, y, depth, 
                          cmap='jet',
                          linewidth=0)
    
    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    
    # 设置标题
    plt.title('3D Depth Visualization')
    
    return fig

def visualize_depth(depth_data):
    """
    将深度数据可视化为彩色图像，nan值显示为黑色
    
    参数:
        depth_data: 深度图像数据
    
    返回:
        彩色深度图和对应的归一化深度数据
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
    normalized_uint8 = (normalized * 255).astype(np.uint8)
    
    # 应用色彩映射
    depth_colormap = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
    
    # 将nan对应的像素设置为黑色 [0,0,0]
    depth_colormap[nan_mask] = [0,0,0]
    
    return depth_colormap, normalized

depth = np.load("/media/clp/Data1/workspace/DPSNet/result/0000.npy")

# 显示2D深度图
depth_colormap, normalized_depth = visualize_depth(depth)

# 使用matplotlib显示带colorbar的2D深度图
plt.figure(figsize=(10, 8))
im = plt.imshow(normalized_depth, cmap='jet')
plt.colorbar(im)
plt.title('2D Depth Visualization')
plt.show()

# 显示3D深度图
fig = visualize_depth_3d(depth)
plt.show()