import numpy as np
import cv2
import torch
import torch.nn.functional as F

class PlaneSweepingStereo:
    def __init__(self, fx, fy, cx, cy, num_planes=64, min_depth=0.5, max_depth=5.0):
        """
        初始化参数
        fx, fy: 相机内参焦距
        cx, cy: 相机主点
        num_planes: 虚拟平面数量
        min_depth, max_depth: 深度范围
        """
        self.K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)
        self.num_planes = num_planes
        self.min_depth = min_depth
        
        # 按照公式生成虚拟平面的深度值
        self.depths = []
        for l in range(1, num_planes + 1):
            dl = (num_planes * min_depth) / l
            self.depths.append(dl)
        self.depths = np.array(self.depths)
        
        # 相机变换 (baseline = 20cm)
        self.T = np.array([0.2, 0, 0])
        self.R = np.eye(3)

    def warp_image(self, source_image, depth):
        """将source image按照给定深度warp到reference视角"""
        h, w = source_image.shape[:2]
        
        # 生成像素网格
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
        
        # 反投影到3D空间
        points_3d = self.K_inv @ pixels.reshape(-1, 3).T
        points_3d *= depth.reshape(-1, 1)
        
        # 将3D点变换到source frame
        points_3d = self.R @ points_3d + self.T.reshape(3, 1)
        
        # 投影回2D
        points_2d = self.K @ points_3d
        points_2d = points_2d[:2] / points_2d[2]
        points_2d = points_2d.T.reshape(h, w, 2)
        
        # 使用pytorch进行双线性插值
        source_image_torch = torch.from_numpy(source_image).float().permute(2, 0, 1).unsqueeze(0)
        points_2d_torch = torch.from_numpy(points_2d).float()
        
        # 归一化坐标到[-1, 1]
        points_2d_norm = torch.stack([
            2 * points_2d_torch[..., 0] / (w - 1) - 1,
            2 * points_2d_torch[..., 1] / (h - 1) - 1
        ], dim=-1)
        
        # 进行网格采样
        warped_image = F.grid_sample(
            source_image_torch,
            points_2d_norm.unsqueeze(0),
            mode='bilinear',
            padding_mode='zeros'
        )
        
        return warped_image.squeeze(0).permute(1, 2, 0).numpy()

    def build_cost_volume(self, ref_image, source_image):
        """构建cost volume"""
        h, w = ref_image.shape[:2]
        cost_volume = np.zeros((h, w, self.num_planes))
        
        # 归一化图像值到[0,1]
        ref_image = ref_image.astype(np.float32) / 255.0
        source_image = source_image.astype(np.float32) / 255.0
        
        for i, depth in enumerate(self.depths):
            # 对每个深度假设进行warping
            warped_image = self.warp_image(source_image, depth)
            
            # 计算光度误差 (可以选择不同的误差度量)
            # 这里使用L1距离
            cost = np.abs(ref_image - warped_image).mean(axis=2)
            cost_volume[..., i] = cost
            
        return cost_volume

    def depth_regression(self, cost_volume):
        """深度回归"""
        costs = torch.from_numpy(cost_volume).float()
        probs = F.softmax(-costs, dim=-1)  # 将cost转换为概率，注意负号
        
        depth_hypotheses = torch.from_numpy(self.depths).float()
        pred_depth = torch.sum(probs * depth_hypotheses.view(1, 1, -1), dim=-1)
        
        return pred_depth.numpy()

    def estimate_depth(self, ref_image, source_image):
        """主函数：估计深度图"""
        # 1. 构建cost volume (直接使用图像强度)
        cost_volume = self.build_cost_volume(ref_image, source_image)
        
        # 2. 深度回归
        depth_map = self.depth_regression(cost_volume)
        np.save("cost_volume.npy", cost_volume)
        np.save("depth_image.npy", depth_map)
        return depth_map

# 使用示例
if __name__ == "__main__":
    # 相机参数
    fx, fy = 360, 360  # 焦距
    cx, cy = 360, 240  # 主点
    
    # 初始化
    pss = PlaneSweepingStereo(fx, fy, cx, cy)
    
    # 读取图像
    ref_image = cv2.imread('/home/clp/catkin_ws/src/sonar_cam_stereo/src/data/rgb_ref/rgb_0.png')
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    
    sonar = np.load('/home/clp/catkin_ws/src/sonar_cam_stereo/src/data/sonar/sonar_0.npy')
    # source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    
    
    cv2.imshow('Camera', ref_image)
    cv2.imshow('Sonar', np.flip(sonar, 0))
    cv2.waitKey(0)