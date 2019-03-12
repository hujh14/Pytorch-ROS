#!/usr/bin/env python
import json
import numpy as np

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image  # the rostopic message we subscribe/publish 
from cv_bridge import CvBridge # package to convert rosmsg<->cv2 

from detector import YOLODetector
from visualize import vis_image

class YOLODetectorNode:

    def __init__(self):
        self.detector = YOLODetector()

        self.pub = rospy.Publisher('/bboxes', String, queue_size=1)
        self.pub_debug = rospy.Publisher('/bboxes/debug', Image, queue_size=1)
        self.sub_image = rospy.Subscriber("/usb_cam/image_raw", Image, self.process_image, queue_size=1)
        self.bridge = CvBridge()

        rospy.loginfo("YOLO node initialized")

    def process_image(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg)
        labels, bboxes = self.detector.predict_image(img)
        output = json.dumps({"labels": labels, "bboxes": bboxes})
        self.pub.publish(output)

        debug_img = vis_image(img, labels, bboxes)
        img_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        self.pub_debug.publish(img_msg)


if __name__ == '__main__':
    rospy.init_node('yolo_pytorch')
    node = YOLODetectorNode()
    rospy.spin()

