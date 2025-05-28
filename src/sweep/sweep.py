import numpy as np
import cv2, os
from load import DataLoader
from scipy.spatial.transform import Rotation as R


def get_rpy_from_matrix(matrix):
    # 确保输入是一个4x4的转移矩阵
    if matrix.shape != (4, 4):
        raise ValueError("输入必须是4x4的转移矩阵")
    
    # 提取3x3旋转矩阵
    rotation_matrix = matrix[:3, :3]
    
    # 使用scipy的Rotation类将旋转矩阵转换为欧拉角
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)
    
    # 返回roll, pitch, yaw (单位：度)
    return euler_angles

def warp(sonar_rect, depth, phi, pitch):
    """
    将原图像通过给定的投影关系变换到目标图像
    
    参数:
    sonar_rect: 原始图像 (numpy数组)
    depth: 深度值
    phi: 垂直视角（弧度）
    pitch: 俯仰角（弧度）
    
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
        d = depth*np.tan(pitch)/(b+np.tan(pitch))
        
        theta_prime = 8 * (np.rad2deg(theta)+45)
        d_prime = 100 * d
        
        return theta, d, theta_prime, d_prime
    u,v=1,1
    theta, d, theta_prime, d_prime = transform_coordinates(u,v)
    print(u,v,theta,d, theta_prime, d_prime)
        
    u,v=720,1
    theta, d, theta_prime, d_prime = transform_coordinates(u,v)
    print(u,v,theta,d, theta_prime, d_prime)
        
    u,v=1,480
    theta, d, theta_prime, d_prime = transform_coordinates(u,v)
    print(u,v,theta,d, theta_prime, d_prime)
        
    u,v=720,480
    theta, d, theta_prime, d_prime = transform_coordinates(u,v)
    print(u,v,theta,d, theta_prime, d_prime)
    
    u,v=360,0
    theta, d, theta_prime, d_prime = transform_coordinates(u,v)
    print(u,v,theta,d, theta_prime, d_prime)
    
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

# 示例使用
def main():
    # 替换为你的数据目录路径
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/data"
    
    # 创建数据加载器
    loader = DataLoader(data_path)
    
    # 读取第一帧数据
    rgb, depth, pose, sonar, sonar_rect = loader.load_frame(0)
    
    # 打印数据信息
    print("RGB image shape:", rgb.shape)
    print("Depth image shape:", depth.shape)
    print("Pose matrix:\n", pose)
    
    print("Sonar image shape:", sonar.shape)
    print("sonar_rect image shape:", sonar_rect.shape)
    
    
    rpy = get_rpy_from_matrix(pose)
    print(f"Roll (度): {rpy[0]:.2f}")
    print(f"Pitch (度): {rpy[1]:.2f}")
    print(f"Yaw (度): {rpy[2]:.2f}")
    
    # 定义变换参数
    depth = 3  # 深度值
    phi = 0.5 * 67.38 * np.pi/180  # 水平视角
    pitch = 15 * np.pi/180  # 俯仰角（示例值）
    
    # 执行变换
    transformed_image = warp(sonar_rect, depth, phi, pitch)
    
    # 显示结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('sonar_rect')
    plt.imshow(sonar_rect)
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('transformed_image')
    plt.imshow(transformed_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    result_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("transformed_image.jpg", result_rgb)
    print("变换后的图像已保存")

if __name__ == "__main__":
    main()