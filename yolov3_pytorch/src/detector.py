from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.insert(0, '../PyTorch-YOLOv3')

import os
import time
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm

from models import Darknet
from utils.utils import *
from utils.datasets import ImageFolder, prepare_image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from visualize import vis_image


class YOLODetector:
    def __init__(self):
        self.img_size = 416
        self.config_path = '../PyTorch-YOLOv3/config/yolov3.cfg'
        self.weights_path = '../PyTorch-YOLOv3/weights/yolov3.weights'
        self.class_path = '../PyTorch-YOLOv3/data/coco.names'
        self.classes = load_classes(self.class_path)
        self.conf_thres = 0.8
        self.nms_thres = 0.4

        self.setup()

    def setup(self):
        cuda = torch.cuda.is_available()

        # Set up model
        model = Darknet(self.config_path, img_size=self.img_size)
        model.load_weights(self.weights_path)
        if cuda:
            model.cuda()
        model.eval() # Set in evaluation mode

        self.model = model
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def predict(self, img):
        img_size = self.img_size

        # Configure input
        input_img = prepare_image(img, (img_size, img_size))
        input_imgs = input_img.unsqueeze(0)
        input_imgs = Variable(input_imgs.type(self.Tensor))

        # Get detections
        detections = []
        with torch.no_grad():
            detections = self.model(input_imgs)
            detections = non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)[0]
            if detections is not None:
                detections = detections.cpu()

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        prediction = []
        labels = []
        bboxes = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            det = {}
            det["category"] = self.classes[int(cls_pred)]
            det["bbox"] = [x1, y1, box_w, box_h]
            det["score"] = cls_conf.item()
            prediction.append(det)
        return prediction

    def visualize(self, img, prediction):
        labels = []
        bboxes = []
        for det in prediction:
            label = "{} {}".format(det["category"], det["score"])
            labels.append(label)
            bboxes.append(det["bbox"])

        debug_image = vis_image(img, labels, bboxes)
        return debug_image


def run_images(im_list, output_dir):
    detector = YOLODetector()

    print("Inference")
    cache = []
    for img_path in tqdm(im_list):
        img = cv2.imread(img_path)[:,:,::-1]
        prediction = detector.predict(img)
        cache.append((img_path, img, prediction))

    print("Visualize")
    for img_path, img, prediction in tqdm(cache):
        out_fn = os.path.join(output_dir, os.path.basename(img_path))
        if not os.path.exists(os.path.dirname(out_fn)):
            os.makedirs(os.path.dirname(out_fn))

        debug_image = detector.visualize(img, prediction)
        cv2.imwrite(out_fn, debug_image[:,:,::-1])

def run_video(video_path, output_dir):
    detector = YOLODetector()

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

        prediction = detector.predict(img)
        cache.append((img, prediction))

    print("Visualize")
    # Prepare output video
    out_fn = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_fn, fourcc, fps, (frame_width, frame_height))
    for img, prediction in tqdm(cache):
        debug_image = lanenet.visualize(img, prediction)
        out.write(debug_image)

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str, help='The image directory')
    parser.add_argument('-v', '--video_path', type=str, help='The video path')
    parser.add_argument('-o', '--output_dir', type=str, help='The output dir', default="./output")
    args = parser.parse_args()

    if args.image_dir:
        im_list = sorted(glob.glob('%s/*.*' % args.image_dir))
        run_images(im_list, args.output_dir)
    elif args.video_path:
        run_video(args.video_path, args.output_dir)
    else:
        print("Choose video or image directory like", '../PyTorch-YOLOv3/data/samples')

