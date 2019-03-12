from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.insert(0, './lanenet-lane-detection')

"""
LaneNet
"""
import os
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
import glog as log
from tqdm import tqdm

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
            binary_seg_image, instance_seg_image = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],
                                                            feed_dict={self.input_tensor: [image]})
            prediction = binary_seg_image, instance_seg_image
            return prediction


    def visualize(self, img, prediction):
        binary_seg_image, instance_seg_image = prediction
        binary_seg_image[0] = self.postprocessor.postprocess(binary_seg_image[0])
        mask_image = self.cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])

        # binary_image = np.array(binary_seg_image[0]*255, dtype='uint8')
        mask_image = cv2.resize(mask_image, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        debug_image = cv2.addWeighted(img, 1, mask_image, 0.6, 0)
        return debug_image


def run_image(image_path, output_dir):
    lanenet = LaneNet()

    img = cv2.imread(image_path)
    out_fn = os.path.join(output_dir, os.path.basename(image_path))

    prediction = lanenet.predict(img)
    debug_image = lanenet.visualize(img, prediction)

    print("Writing to", out_fn)
    cv2.imwrite(out_fn, debug_image)

    lanenet.sess.close()

def run_video(video_path, output_dir):
    lanenet = LaneNet()

    # Load input video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Inference")
    cache = []
    for i in tqdm(range(length)):
        ret, img = cap.read()
        if img is None:
            break

        prediction = lanenet.predict(img)
        cache.append((img, prediction))

    print("Visualize")
    # Prepare output video
    out_fn = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_fn, fourcc, fps, (frame_width, frame_height))
    for i in tqdm(range(len(cache))):
        img, prediction = cache[i]
        debug_image = lanenet.visualize(img, prediction)
        out.write(debug_image)

    cap.release()
    lanenet.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, help='The image path')
    parser.add_argument('-v', '--video_path', type=str, help='The video path')
    parser.add_argument('-o', '--output_dir', type=str, help='The output dir', default="./output")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.image_path:
        run_image(args.image_path, args.output_dir)
    elif args.video_path:
        run_video(args.video_path, args.output_dir)
    else:
        print("Choose image or video.")




