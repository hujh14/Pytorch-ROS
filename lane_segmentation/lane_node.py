#!/usr/bin/env python
import json
import numpy as np

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image  # the rostopic message we subscribe/publish 
from cv_bridge import CvBridge # package to convert rosmsg<->cv2 

from lanenet import LaneNet
from lane_detection_cv import detect_lanes

class LaneDetectorNode:

    def __init__(self):
        self.use_lanenet = True
        if self.use_lanenet:
            self.detector = LaneNet()

        self.pub = rospy.Publisher('/lanes', String, queue_size=1)
        self.pub_debug = rospy.Publisher('/lanes/debug', Image, queue_size=1)
        self.sub_image = rospy.Subscriber("/usb_cam/image_raw", Image, self.process_image, queue_size=1)
        self.bridge = CvBridge()

        rospy.loginfo("LaneNet node initialized")

    def process_image(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg)

        if self.use_lanenet:
            mask_image, debug_image = self.detector.predict(img)
        else:
            lanes, debug_image = detect_lanes(img)
            output = json.dumps({"lanes": lanes.tolist()})
            self.pub.publish(output)

        img_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
        self.pub_debug.publish(img_msg)


if __name__ == '__main__':
    rospy.init_node('LaneNet')
    node = LaneDetectorNode()
    rospy.spin()

