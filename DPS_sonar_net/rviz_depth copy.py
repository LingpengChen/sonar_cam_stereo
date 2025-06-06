#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import struct
import tf
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

class DepthPublisher:
    def __init__(self, num_publishers):
        # Initialize ROS node
        rospy.init_node('depth_publisher', anonymous=True)
        
        # Create multiple publishers
        self.pubs = {}
        self.tfs = {}
        for i in range(num_publishers):
            topic_name = f'/depth_cloud_{i}'
            frame_name = f'camera_link_{i}'
            self.pubs[i] = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
            self.tfs[i] = frame_name
        
        # Create TF broadcaster
        self.broadcaster = StaticTransformBroadcaster()
        
        # Publish static TF transformations
        self.publish_static_tfs()
        
        # Set publishing rate
        self.rate = rospy.Rate(10)  # 10Hz

    # def publish_static_tfs(self):
    #     """
    #     Publish static TF transformations from map to each camera_link
    #     """
    #     static_transforms = []
        
    #     for i, frame_id in self.tfs.items():
    #         static_transformStamped = TransformStamped()
    #         static_transformStamped.header.stamp = rospy.Time.now()
    #         static_transformStamped.header.frame_id = "map"
    #         static_transformStamped.child_frame_id = frame_id

    #         # You can modify these values to set different positions for each camera
    #         static_transformStamped.transform.translation.x = float(i)  # Offset each camera by 1 unit in x
    #         static_transformStamped.transform.translation.y = 0.0
    #         static_transformStamped.transform.translation.z = 0.0

    #         static_transformStamped.transform.rotation.x = 0.0
    #         static_transformStamped.transform.rotation.y = 0.0
    #         static_transformStamped.transform.rotation.z = 0.0
    #         static_transformStamped.transform.rotation.w = 1.0

    #         static_transforms.append(static_transformStamped)

    #     self.broadcaster.sendTransform(static_transforms)

    def publish_static_tfs(self):
        """
        Publish static TF transformations:
        1. map -> base_link_{i} (robot base frame)
        2. base_link_{i} -> camera_link_{i} (camera frame)
        """
        static_transforms = []
        
        for i, frame_id in self.tfs.items():
            # First transform: map -> base_link
            base_frame = f'base_link_{i}'
            map_to_base = TransformStamped()
            map_to_base.header.stamp = rospy.Time.now()
            map_to_base.header.frame_id = "map"
            map_to_base.child_frame_id = base_frame

            # Set robot base position (可以根据需要修改)
            # map_to_base.transform.translation.x = float(i) * 2.0  # 机器人间隔2米
            map_to_base.transform.translation.x = 0.0  # 机器人间隔2米
            map_to_base.transform.translation.y = 0.0
            map_to_base.transform.translation.z = 2

            # Robot orientation in map frame (可以根据需要修改)
            # map_to_base.transform.rotation.x = 0.0
            # map_to_base.transform.rotation.y = 0.0
            # map_to_base.transform.rotation.z = 0.0
            # map_to_base.transform.rotation.w = 1.0
            # 相机的z轴对齐机器人的x轴（90度绕x轴旋转）
            map_to_base.transform.rotation.x = np.sin(-np.pi/12)  # sin(pi/4)
            map_to_base.transform.rotation.y = 0.0
            map_to_base.transform.rotation.z = 0.0
            map_to_base.transform.rotation.w = np.cos(-np.pi/12)  # cos(pi/4)
            
            
            static_transforms.append(map_to_base)

            # Second transform: base_link -> camera_link
            base_to_camera = TransformStamped()
            base_to_camera.header.stamp = rospy.Time.now()
            base_to_camera.header.frame_id = base_frame
            base_to_camera.child_frame_id = frame_id

            # Camera position relative to robot base (可以根据需要修改)
            base_to_camera.transform.translation.x = 0.0  # 相机在机器人前方0.3米
            base_to_camera.transform.translation.y = 0.0
            base_to_camera.transform.translation.z = 0.5  # 相机高度0.5米

            # Convert camera frame to align with robot frame
            # 相机坐标系到机器人坐标系的旋转
            # 相机的z轴对齐机器人的x轴（-90度绕y轴旋转）
            # base_to_camera.transform.rotation.x = 0.0
            # base_to_camera.transform.rotation.y = -0.7071068  # sin(-pi/4)
            # base_to_camera.transform.rotation.z = 0.0
            # base_to_camera.transform.rotation.w = 0.7071068   # cos(-pi/4)
            
            # 相机的z轴对齐机器人的x轴（90度绕x轴旋转）
            base_to_camera.transform.rotation.x = -0.7071068  # sin(pi/4)
            base_to_camera.transform.rotation.y = 0.0
            base_to_camera.transform.rotation.z = 0.0
            base_to_camera.transform.rotation.w = 0.7071068  # cos(pi/4)
            
            static_transforms.append(base_to_camera)

        self.broadcaster.sendTransform(static_transforms)

    def depth_to_pointcloud(self, depth_data, frame_id):
        """
        Convert depth image to point cloud data using NumPy vectorized operations
        """
        rows, cols = depth_data.shape
        
        # Create grid coordinates
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Camera intrinsics
        fx = 525.0
        fy = 525.0
        cx = cols/2
        cy = rows/2

        # Create point cloud message header
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        # Define point cloud fields
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ]

        # Create valid depth mask
        valid_mask = ~np.isnan(depth_data) & (depth_data > 0)
        
        # Calculate 3D coordinates
        x = np.where(valid_mask, (u - cx) * depth_data / fx, 0)
        y = np.where(valid_mask, (v - cy) * depth_data / fy, 0)
        z = np.where(valid_mask, depth_data, 0)

        # Calculate color
        normalized_depth = np.clip(depth_data / np.nanmax(depth_data), 0, 1)
        
        # Create RGB color array
        colors = np.zeros(depth_data.shape, dtype=np.uint32)
        valid_colors = (normalized_depth[valid_mask] * 255).astype(np.uint8)
        
        # Use bit operations to create RGBA
        rgb_packed = (valid_colors.astype(np.uint32) << 16 |
                    np.zeros_like(valid_colors, dtype=np.uint32) << 8 |
                    ((255 - valid_colors).astype(np.uint32)) |
                    (np.full_like(valid_colors, 255, dtype=np.uint32) << 24))
        
        # Extract valid points' coordinates and colors
        points = np.column_stack((
            x[valid_mask],
            y[valid_mask],
            z[valid_mask],
            rgb_packed
        ))

        # Create PointCloud2 message
        pc2_msg = pc2.create_cloud(header, fields, points)
        return pc2_msg
    
    def publish_depths(self, depth_data_dict):
        """
        Publish multiple depth data
        
        Parameters:
            depth_data_dict: Dictionary mapping publisher index to depth image data
        """
        for idx, depth_data in depth_data_dict.items():
            if idx in self.pubs:
                pc2_msg = self.depth_to_pointcloud(depth_data, self.tfs[idx])
                self.pubs[idx].publish(pc2_msg)

def main():
    try:
        # Create publisher object with the number of publishers you need
        num_publishers = 2  # Change this number based on how many point clouds you want to publish
        depth_publisher = DepthPublisher(num_publishers)
        
        # Load depth images
        est_depth = np.load("/home/clp/catkin_ws/src/sonar_cam_stereo/DPS_sonar_net/result/0003.npy")
        true_depth = np.load("/home/clp/catkin_ws/src/sonar_cam_stereo/DPS_sonar_net/result/0003_gt.npy")
        
        # Create dictionary mapping publisher index to depth data
        depth_data_dict = {
            0: est_depth,
            1: true_depth
        }
        
        # Continuously publish data
        while not rospy.is_shutdown():
            depth_publisher.publish_depths(depth_data_dict)
            depth_publisher.rate.sleep()

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()