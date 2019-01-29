from __future__ import division

import os
import sys
sys.path.insert(0, '../PyTorch-YOLOv3')
import time
import argparse
import glob
import cv2
import numpy as np

from models import Darknet
from utils.utils import *
from utils.datasets import ImageFolder, prepare_image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from visualize import vis_image

class YOLODetector:
    def __init__(self, opt):
        cuda = torch.cuda.is_available() and opt.use_cuda
        # Set up model
        model = Darknet(opt.config_path, img_size=opt.img_size)
        model.load_weights(opt.weights_path)
        if cuda:
            model.cuda()
        model.eval() # Set in evaluation mode

        self.model = model
        self.opt = opt
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def predict(self, input_imgs):
        # Configure input
        input_imgs = Variable(input_imgs.type(self.Tensor))
        # Get detections
        with torch.no_grad():
            detections = self.model(input_imgs)
            detections = non_max_suppression(detections, 80, self.opt.conf_thres, self.opt.nms_thres)
            return detections

    def predict_image(self, img):
        img_size = self.opt.img_size
        classes = load_classes(self.opt.class_path) # Extracts class labels from file

        input_img = prepare_image(img, (img_size, img_size))
        input_imgs = input_img.unsqueeze(0)
        detections = self.predict(input_imgs)[0]
        if detections is None:
            detections = []

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        labels = []
        bboxes = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            label = classes[int(cls_pred)]
            bbox = np.array([x1, y1, box_w, box_h])
            # print('\t+ Label: %s, Conf: %.5f' % (label, cls_conf.item()))
            labels.append(label)
            bboxes.append(bbox.tolist())
        return labels, bboxes


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='../PyTorch-YOLOv3/data/samples', help='path to dataset')
    parser.add_argument('--config_path', type=str, default='../PyTorch-YOLOv3/config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='../PyTorch-YOLOv3/weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='../PyTorch-YOLOv3/data/coco.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == "__main__":
    opt = get_opt()
    detector = YOLODetector(opt)

    img_paths = sorted(glob.glob('%s/*.*' % opt.image_folder))
    for img_path in img_paths:
        print(img_path)
        img = cv2.imread(img_path)[:,:,::-1]
        labels, bboxes = detector.predict_image(img)
        img_vis = vis_image(img, labels, bboxes)

        output_path = "output/" + os.path.basename(img_path)
        cv2.imwrite(output_path, img_vis[:,:,::-1])


