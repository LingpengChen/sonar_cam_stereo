import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取原始图像
ref_image = cv2.imread('/home/clp/catkin_ws/src/sonar_cam_stereo/src/data/rgb_ref/rgb_0.png')
ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

# 定义相机内参
fx, fy = 360, 360  # 焦距
cx, cy = 360, 240  # 主点
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# 定义第一个相机的外参矩阵（位于(5,0,0)，x朝向原点）

# 如果相机朝向平面(x=0)，第一个相机的旋转矩阵应该是：
R1 = np.array([
    [0, 0, -1],
    [1, 0,  0],
    [0, -1, 0]
])
t1 = np.array([5, 0, 0])

T1 = np.eye(4)
T1[:3, :3] = R1
T1[:3, 3] = t1

# 定义第二个相机的外参矩阵（位于(5/√2,0,5/√2)，x朝向原点）

# 第二个相机的旋转矩阵应该是：
R2 = np.array([
    [0,  1/np.sqrt(2),       -1/np.sqrt(2)],
    [1,  0,                     0],
    [0,  -1/np.sqrt(2),     -1/np.sqrt(2)]
])
t2 = np.array([5/np.sqrt(2), 0, 5/np.sqrt(2)])
T2 = np.eye(4)
T2[:3, :3] = R2
T2[:3, 3] = t2

# 可视化相机位姿
def plot_camera_pose(ax, R, t, color='b'):
    # 绘制相机中心
    ax.scatter(t[0], t[1], t[2], c=color, marker='o')
    
    # 绘制相机坐标轴
    axis_length = 1
    for i, c in enumerate(['r', 'g', 'b']):
        axis = R[:, i] * axis_length
        ax.quiver(t[0], t[1], t[2],
                 axis[0], axis[1], axis[2],
                 color=c)
    
    # 绘制视线方向（指向原点）
    sight_line = -t / np.linalg.norm(t) * 2
    ax.quiver(t[0], t[1], t[2],
             sight_line[0], sight_line[1], sight_line[2],
             color='k', linestyle='--')

# 创建3D图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制两个相机的位姿
plot_camera_pose(ax, R1, t1, 'blue')
plot_camera_pose(ax, R2, t2, 'red')

# 绘制平面 x=0
xx, yy = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
zz = np.zeros_like(xx)
ax.plot_surface(np.zeros_like(xx), xx, yy, alpha=0.2)

# 设置坐标轴标签和范围
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-6, 2])
ax.set_ylim([-4, 4])
ax.set_zlim([-6, 2])

# 添加图例
ax.text(t1[0], t1[1], t1[2], 'Camera 1', color='blue')
ax.text(t2[0], t2[1], t2[2], 'Camera 2', color='red')

# plt.title('Camera Poses Visualization')
# plt.show()

# 计算相对变换
T_rel = np.linalg.inv(T2) @ T1
R = T_rel[:3, :3]
t = T_rel[:3, 3].reshape(3,1)

# 计算平面法向量（在世界坐标系中）
n = np.array([0, 0, 1]).reshape(3,1)  # 平面法向量
d = 5  # 平面到原点的距离

# 计算单应矩阵
print(R)
print(t)
H = K @ (R + (t @ n.T)/d) @ np.linalg.inv(K)
print(H)
# 归一化单应矩阵
H = H / H[2,2]

# 3. 构建平移矩阵,将负坐标区域移到可见区域
# min = 400
# T = np.array([
#     [1, 0, min],  # 如果x_min为负,则向右平移|x_min|个像素
#     [0, 1, min],  # 如果y_min为负,则向下平移|y_min|个像素
#     [0, 0, 1]
# ])

# 4. 计算新的变换矩阵
# H = T @ H

# 应用单应变换
# warped_image = cv2.warpPerspective(ref_image, H, (2*ref_image.shape[1], 2*ref_image.shape[0]))
warped_image = cv2.warpPerspective(ref_image, H, (ref_image.shape[1], ref_image.shape[0]))


# 显示结果
cv2.imshow('Original Image', ref_image)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()