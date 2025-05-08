import os
import cv2
import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        # 设置数据目录
        self.base_dir = data_dir
        self.rgb_dir = os.path.join(self.base_dir, "rgb_ref")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.pose_dir = os.path.join(self.base_dir, "pose")
        self.sonar_dir = os.path.join(self.base_dir, "sonar")
        self.sonar_rect_dir = os.path.join(self.base_dir, "sonar_rect")
        
        # 获取所有文件列表
        self.rgb_files = sorted(os.listdir(self.rgb_dir))
        self.depth_files = sorted(os.listdir(self.depth_dir))
        self.pose_files = sorted(os.listdir(self.pose_dir))
        self.sonar_files = sorted(os.listdir(self.sonar_dir))
        self.sonar_rect_files = sorted(os.listdir(self.sonar_rect_dir))
        
        # 检查文件数量是否匹配
        assert len(self.rgb_files) == len(self.depth_files) == len(self.pose_files) == len(self.sonar_files), \
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
      
        sonar_path = os.path.join(self.sonar_dir, self.sonar_files[idx])
        sonar = np.load(sonar_path)
        
        sonar_rect_path = os.path.join(self.sonar_rect_dir, self.sonar_rect_files[idx])
        sonar_rect = np.load(sonar_rect_path)
        
        return rgb, depth, pose, sonar, sonar_rect

    def __len__(self):
        return self.num_frames

# 使用示例
if __name__ == "__main__":
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
    
    # 可视化RGB图像
    # cv2.imshow("rgb", rgb)
    # cv2.imshow("sonar", np.flip(sonar, 0))
    # cv2.imshow("sonar_rect", np.flip(sonar_rect, 0))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("/home/clp/catkin_ws/src/sonar_cam_stereo/src/data/jpg/gray.jpg", gray)
    
    # 可视化深度图（归一化显示）
    from depth2sonar import visualize_depth
    cv2.imshow("Depth", visualize_depth(depth))
    cv2.waitKey(0)
    cv2.destroyAllWindows()