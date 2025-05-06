import os
import cv2
import numpy as np

    
# 使用示例
if __name__ == "__main__":
    # 替换为你的数据目录路径
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/data"
    
    depth = np.load("/home/clp/catkin_ws/src/sonar_cam_stereo/src/depth_image.npy")
    
    # 可视化RGB图像
    cv2.imshow("depth", depth)
    
    # 可视化深度图（归一化显示）
    from depth2sonar import visualize_depth
    cv2.imshow("Depth", visualize_depth(depth))
    cv2.waitKey(0)
    cv2.destroyAllWindows()