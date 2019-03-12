#!/usr/bin/env python
import cv2
import numpy as np

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image  # the rostopic message we subscribe/publish 
from cv_bridge import CvBridge # package to convert rosmsg<->cv2 

class VideoPlayer:

    def __init__(self):
        self.pub = rospy.Publisher('/usb_cam/image_raw', Image, queue_size=1)
        self.bridge = CvBridge()

        # video_path = "../test_videos/dashcam_video.mp4"
        video_path = "../test_videos/highway.mp4"
        self.cap = cv2.VideoCapture(video_path)

        rospy.loginfo("VideoPlayer initialized")

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            rospy.loginfo("Restarted video")
            ret, frame = self.cap.read()

        image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.pub.publish(image_msg)

if __name__ == '__main__':
    rospy.init_node('VideoPlayer')
    node = VideoPlayer()

    r = rospy.Rate(30)
    while not rospy.is_shutdown():
       node.publish_frame()
       r.sleep()


