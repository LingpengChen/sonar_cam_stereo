#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

def image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg)
    print("Image shape:", cv_image.shape)
    cv2.imshow("test", cv_image)
    cv2.waitKey(1)  # 正确的拼写

def main():
    rospy.init_node('image_shape_node')
    rospy.Subscriber('/oculus/postprocess/drawn_sonar', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()