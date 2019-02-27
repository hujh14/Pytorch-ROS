#!/usr/bin/env python
import json
import numpy as np

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image  # the rostopic message we subscribe/publish 
from cv_bridge import CvBridge # package to convert rosmsg<->cv2 

from lanenet import LaneNet

class LaneDetectorNode:

    def __init__(self):
        self.detector = LaneNet()

        self.pub = rospy.Publisher('/lanes', String, queue_size=1)
        self.pub_debug = rospy.Publisher('/lanes/debug', Image, queue_size=1)
        self.sub_image = rospy.Subscriber("/usb_cam/image_raw", Image, self.process_image, queue_size=1)
        self.bridge = CvBridge()

        rospy.loginfo("LaneNet node initialized")

    def process_image(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg)
        lanes, lanes_debug = self.detector.predict(img)
        output = json.dumps({"lanes": lanes})
        self.pub.publish(output)

        img_msg = self.bridge.cv2_to_imgmsg(lanes_debug, "bgr8")
        self.pub_debug.publish(img_msg)


if __name__ == '__main__':
    rospy.init_node('LaneNet')
    node = LaneDetectorNode()
    rospy.spin()

