#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import PointCloud2, PointField, Image
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import struct
import tf
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from pathlib import Path
import time

class DataPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('data_publisher', anonymous=True)
        
        # Create publishers for point clouds
        self.pubs = {
            'est': rospy.Publisher('/depth_cloud_est', PointCloud2, queue_size=1),
            'gt': rospy.Publisher('/depth_cloud_gt', PointCloud2, queue_size=1)
        }
        
        # Create publishers for images
        self.image_pubs = {
            'rgb': rospy.Publisher('/rgb_image', Image, queue_size=1),
            'sonar': rospy.Publisher('/sonar_image', Image, queue_size=1)
        }
        
        # Create frames dictionary
        self.tfs = {
            'est': 'camera_link_est',
            'gt': 'camera_link_gt'
        }
        
        # Create bridge for converting OpenCV images to ROS messages
        self.bridge = CvBridge()
        
        # Create TF broadcaster
        self.broadcaster = StaticTransformBroadcaster()
        
        # Publish static TF transformations
        self.publish_static_tfs()
        
        # Set publishing rate
        self.rate = rospy.Rate(1)  # 1Hz, 可以根据需要调整

    def publish_static_tfs(self):
        """
        Publish static TF transformations for both estimated and ground truth frames
        """
        static_transforms = []
        
        for name, frame_id in self.tfs.items():
            # Transform from map to camera link
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "map"
            transform.child_frame_id = frame_id

            # Set position
            transform.transform.translation.x = 0.0
            transform.transform.translation.y = 0.0 
            transform.transform.translation.z = 1.0 if name == 'est' else 3.0  # GT相机往z轴偏移2米

            # Set rotation
            transform.transform.rotation.x = np.sin(-np.pi/3.5)
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = np.cos(-np.pi/3.5)
            
            static_transforms.append(transform)

        self.broadcaster.sendTransform(static_transforms)

    def depth_to_pointcloud(self, depth_data, frame_id):
        """
        Convert depth image to point cloud data
        """
        rows, cols = depth_data.shape
        
        # Create grid coordinates
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Camera intrinsics
        fx = 525.0
        fy = 525.0
        cx = cols/2
        cy = rows/2

        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ]

        valid_mask = ~np.isnan(depth_data) & (depth_data > 0)
        
        x = np.where(valid_mask, (u - cx) * depth_data / fx, 0)
        y = np.where(valid_mask, (v - cy) * depth_data / fy, 0)
        z = np.where(valid_mask, depth_data, 0)

        normalized_depth = np.clip(depth_data / np.nanmax(depth_data), 0, 1)
        
        colors = np.zeros(depth_data.shape, dtype=np.uint32)
        valid_colors = (normalized_depth[valid_mask] * 255).astype(np.uint8)
        
        rgb_packed = (valid_colors.astype(np.uint32) << 16 |
                    np.zeros_like(valid_colors, dtype=np.uint32) << 8 |
                    ((255 - valid_colors).astype(np.uint32)) |
                    (np.full_like(valid_colors, 255, dtype=np.uint32) << 24))
        
        points = np.column_stack((
            x[valid_mask],
            y[valid_mask],
            z[valid_mask],
            rgb_packed
        ))

        return pc2.create_cloud(header, fields, points)

    def publish_data(self, folder_path):
        """
        Publish all data from a specific folder
        """
        # Load depth data
        est_depth = np.load(folder_path / 'output_depth.npy')
        gt_depth = np.load(folder_path / 'gt_depth.npy')
        
        # Load images
        rgb_img = cv2.imread(str(folder_path / 'rgb_image.png'))
        sonar_img = cv2.imread(str(folder_path / 'sonar_image.png'))
        
        # Convert depth data to point clouds and publish
        self.pubs['est'].publish(self.depth_to_pointcloud(est_depth, self.tfs['est']))
        self.pubs['gt'].publish(self.depth_to_pointcloud(gt_depth, self.tfs['gt']))
        
        # Convert images to ROS messages and publish
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, "bgr8")
        sonar_msg = self.bridge.cv2_to_imgmsg(sonar_img, "bgr8")
        
        self.image_pubs['rgb'].publish(rgb_msg)
        self.image_pubs['sonar'].publish(sonar_msg)

def main():
    try:
        # Create publisher object
        data_publisher = DataPublisher()
        
        # Get the result directory path
        result_dir = Path("result_tiny_canyon")  # 修改为你的result目录路径
        
        while not rospy.is_shutdown():
            # Iterate through all index folders
            for folder in sorted(result_dir.glob("[0-9]*")):
                if rospy.is_shutdown():
                    break
                
                # 获取文件夹名称并转换为整数进行比较
                folder_name = folder.name
                if folder_name.isdigit() and int(folder_name) > 800:
                    rospy.loginfo(f"Publishing data from folder: {folder}")
                    data_publisher.publish_data(folder)
                    data_publisher.rate.sleep()
                    # Optional: 添加延迟后重新开始循环
                    time.sleep(2)
            
            
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()