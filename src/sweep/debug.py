import numpy as np
import cv2

ref_image = cv2.imread('/home/clp/catkin_ws/src/sonar_cam_stereo/src/data/rgb_ref/rgb_0.png')
ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

# 定义相机内参(假设两个相机内参相同)
fx, fy = 360, 360  # 焦距
cx, cy = 360, 240  # 主点
K = np.array([[fx, 0, cx],
              [0, fy, cy], 
              [0, 0, 1]])

# 定义cam1的pose (将图像放在z=5m处)
# cam1的位姿为单位矩阵,即位于原点朝向z轴正方向
R1 = np.eye(3)
t1 = np.array([0, 0, 0]).reshape(3,1)
P1 = np.hstack((R1, t1))

# 定义cam2的pose (假设在不同位置观察同一平面)
# 这里假设cam2在x方向平移1m,向上倾斜15度
angle = 45 * np.pi / 180  # 15度转弧度
R = np.array([[1, 0, 0],
               [0, np.cos(angle), -np.sin(angle)],
               [0, np.sin(angle), np.cos(angle)]])
# t2 = np.array([ 0, 0, 0]).reshape(3,1)  # cam2的平移向量
t = np.array([  0, 5/np.sqrt(2), 5-5/np.sqrt(2)]).reshape(3,1)  # cam2的平移向量


# 定义平面法向量和深度 (z=5m的平面)
n = np.array([0, 0, 1]).reshape(3,1)  # 平面法向量
d = 5.0  # 平面到cam1原点的距离

# 计算单应矩阵
print(R)
print(t)

H = K @ (R - (t @ n.T)/d) @ np.linalg.inv(K)
print(H)

# 归一化单应矩阵
H = H / H[2,2]

# 3. 构建平移矩阵,将负坐标区域移到可见区域
min = 1000
T = np.array([
    [1, 0, min],  # 如果x_min为负,则向右平移|x_min|个像素
    [0, 1, 2*min],  # 如果y_min为负,则向下平移|y_min|个像素
    [0, 0, 1]
])

# 4. 计算新的变换矩阵
H = T @ H

# 使用cv2.warpPerspective进行图像变换
warped_image = cv2.warpPerspective(ref_image, H, (4*ref_image.shape[1], 4*ref_image.shape[0]))
# warped_image = cv2.warpPerspective(ref_image, H, (100, 100))

cv2.imshow('Original Image', ref_image)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()