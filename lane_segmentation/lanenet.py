import sys
sys.path.insert(0, './lanenet-lane-detection')

"""
LaneNet
"""
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import glog as log
import matplotlib.pyplot as plt

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

class LaneNet:

    def __init__(self):
        self.use_gpu = 1
        self.weights_path = "lanenet-lane-detection/model/tusimple_lanenet/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000"

        self.setup()

    def setup(self):
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        phase_tensor = tf.constant('test', tf.string)

        net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='lanenet_model')

        self.cluster = lanenet_cluster.LaneNetCluster()
        self.postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

        saver = tf.train.Saver()

        # Set sess configuration
        if self.use_gpu:
            sess_config = tf.ConfigProto(device_count={'GPU': 1})
        else:
            sess_config = tf.ConfigProto(device_count={'CPU': 0})
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=sess_config)

        with self.sess.as_default():
            saver.restore(sess=self.sess, save_path=self.weights_path)

    def predict(self, img):
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image - VGG_MEAN

        with self.sess.as_default():
            t_start = time.time()
            binary_seg_image, instance_seg_image = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],
                                                            feed_dict={self.input_tensor: [image]})
            # print("Interence:", time.time() - t_start)

            binary_seg_image[0] = self.postprocessor.postprocess(binary_seg_image[0])
            mask_image = self.cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                               instance_seg_ret=instance_seg_image[0])

            # binary_image = np.array(binary_seg_image[0]*255, dtype='uint8')
            mask_image = cv2.resize(mask_image, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            debug_image = cv2.addWeighted(img, 1, mask_image, 0.6, 0)
            return mask_image, debug_image

def run_image():
    lanenet = LaneNet()
    image_path = "lanenet-lane-detection/data/tusimple_test_image/0.jpg"
    img = cv2.imread(image_path)

    mask_image, debug_image = lanenet.predict(img)

    cv2.imshow('lanes', debug_image)
    cv2.waitKey(0)

    lanenet.sess.close()

def run_video():
    lanenet = LaneNet()
    video_path = "test_videos/challenge.mp4"

    cap = cv2.VideoCapture(video_path)
    while(True):
        ret, img = cap.read()
        mask_image, debug_image = lanenet.predict(img)

        cv2.imshow('lanes', debug_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    lanenet.sess.close()


if __name__ == '__main__':
    run_image()
    # run_video()
