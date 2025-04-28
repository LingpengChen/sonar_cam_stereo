#!/usr/bin/env python
import rospy
import os
from sensor_msgs.msg import Image
from gazebo_msgs.msg import LinkStates
from cv_bridge import CvBridge
import cv2
import numpy as np
from datetime import datetime
import keyboard

from scipy.spatial.transform import Rotation

# Convert quaternion and position to transformation matrix
def pose_to_matrix(pose):
    # Extract position
    pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    
    # Extract quaternion and convert to rotation matrix
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    rot_matrix = Rotation.from_quat(quat).as_matrix()
    
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = pos
    
    return transform


class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector')
        
        # 创建存储目录
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_rgb_ref_dir = os.path.join(self.base_dir, "data/rgb_ref") # right
        self.save_rgb_source_dir = os.path.join(self.base_dir, "data/rgb_source") # left
        self.save_depth_dir = os.path.join(self.base_dir, "data/depth")
        self.save_pose_dir = os.path.join(self.base_dir, "data/pose")
        
        for dir_path in [self.save_rgb_ref_dir, self.save_rgb_source_dir, self.save_depth_dir, self.save_pose_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # 初始化变量
        self.bridge = CvBridge()
        
        self.current_pose = None
        self.current_rgb_ref = None
        self.current_rgb_source = None
        self.current_depth = None
        
        self.counter = 0

        # 订阅话题
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.pose_callback)
        rospy.Subscriber('/rexrov/rexrov/camera_ref/rgb/image_raw', Image, self.rgb_ref_callback)
        rospy.Subscriber('/rexrov/rexrov/camera_source/rgb/image_raw', Image, self.rgb_source_callback)
        rospy.Subscriber('/rexrov/rexrov/camera_ref/depth/image_raw', Image, self.depth_callback)

    def pose_callback(self, msg):
        try:
            # 找到目标link的索引
            index = msg.name.index('rexrov::rexrov/camera_ref_link')
            self.current_pose = msg.pose[index]
        except ValueError:
            rospy.logwarn("Cannot find camera link in link states")

    def rgb_ref_callback(self, msg):
        self.current_rgb_ref = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def rgb_source_callback(self, msg):
        self.current_rgb_source = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.current_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def save_data(self):
        print(f"Save {self.counter}")
        if (self.current_pose is None or 
            self.current_rgb_ref is None or 
            self.save_rgb_source_dir is None or 
            self.current_depth is None):
            rospy.logwarn("Some data is not ready yet")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(self.current_rgb_ref.shape)
        print(self.current_rgb_source.shape)
        print(self.current_depth.shape)
        print(self.current_pose)
        # 保存RGB图像
        rgb_filename = os.path.join(self.save_rgb_ref_dir, f"rgb_{self.counter}.png")
        cv2.imwrite(rgb_filename, self.current_rgb_ref)
        rgb_filename = os.path.join(self.save_rgb_source_dir, f"rgb_{self.counter}.png")
        cv2.imwrite(rgb_filename, self.current_rgb_source)

        # 保存深度图像
        depth_filename = os.path.join(self.save_depth_dir, f"depth_{self.counter}")
        # np.save(depth_filename, self.current_depth)
        np.savez_compressed(depth_filename, depth=self.current_depth)


        # 保存位姿
        pose_filename = os.path.join(self.save_pose_dir, f"pose_{self.counter}.npy")
        transform_matrix = pose_to_matrix(self.current_pose)
        np.save(pose_filename, transform_matrix)

        self.counter += 1
        rospy.loginfo(f"Saved data set {self.counter} at timestamp {timestamp}")

    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        rospy.loginfo("Press 's' to save data, 'q' to quit")
        
        import sys
        import tty
        import termios
        
        def getch():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        while not rospy.is_shutdown():
            try:
                if sys.stdin.isatty():
                    key = getch()
                    if key == 's':
                        self.save_data()
                        rospy.sleep(0.5)  # 防止多次触发
                    elif key == 'q':
                        break
                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    try:
        collector = DataCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass