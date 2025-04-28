import os
import cv2
import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        # 设置数据目录
        self.base_dir = data_dir
        self.rgb_dir = os.path.join(self.base_dir, "rgb")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.pose_dir = os.path.join(self.base_dir, "pose")
        
        # 获取所有文件列表
        self.rgb_files = sorted(os.listdir(self.rgb_dir))
        self.depth_files = sorted(os.listdir(self.depth_dir))
        self.pose_files = sorted(os.listdir(self.pose_dir))
        
        # 检查文件数量是否匹配
        assert len(self.rgb_files) == len(self.depth_files) == len(self.pose_files), \
            "Number of files in rgb, depth and pose directories do not match!"
        
        self.num_frames = len(self.rgb_files)
        print(f"Found {self.num_frames} frames of data")

    def load_frame(self, idx):
        """加载单帧数据"""
        if idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range, total frames: {self.num_frames}")
        
        # 读取RGB图像
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        rgb = cv2.imread(rgb_path)
        # shape: (480, 720, 3)  dtype('uint8')
        
        # 读取深度图
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth = np.load(depth_path)['depth']
        # (480, 720)  dtype: dtype('float32')
        
        # 读取位姿矩阵
        pose_path = os.path.join(self.pose_dir, self.pose_files[idx])
        pose = np.load(pose_path)
        
        return rgb, depth, pose

    def __len__(self):
        return self.num_frames

# 使用示例
if __name__ == "__main__":
    # 替换为你的数据目录路径
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/data"
    
    # 创建数据加载器
    loader = DataLoader(data_path)
    
    # 读取第一帧数据
    rgb, depth, pose = loader.load_frame(0)
    
    # 打印数据信息
    print("RGB image shape:", rgb.shape)
    print("Depth image shape:", depth.shape)
    print("Pose matrix:\n", pose)
    
    # 可视化RGB图像
    cv2.imshow("RGB", rgb)
    
    # 可视化深度图（归一化显示）
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
    cv2.imshow("Depth", depth_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()