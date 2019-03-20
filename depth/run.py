import sys
sys.path.insert(0, './MegaDepth')

import argparse
import os
import cv2
import numpy as np
from skimage import io
from skimage.transform import resize

import torch
from torch.autograd import Variable
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model

class MegaDepth:

    def __init__(self):

        self.setup()

    def setup(self):
        self.model = create_model(opt)
        self.model.switch_to_eval()

    def preprocess(self, img):
        input_height = 384
        input_width  = 512
        img = np.float32(img)/255.0
        img = resize(img, (input_height, input_width), order = 1)
        img = np.transpose(img, (2,0,1))
        return img

    def predict(self, img):
        img = self.preprocess(img)
        input_img =  torch.from_numpy(img).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.cuda())
        pred_log_depth = self.model.netG.forward(input_images) 
        pred_log_depth = torch.squeeze(pred_log_depth)
        pred_depth = torch.exp(pred_log_depth)
        return pred_depth

    def visualize(self, img, prediction):
        # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
        # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
        pred_inv_depth = 1/prediction
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        # you might also use percentile for better visualization
        pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)
        return pred_inv_depth

        # print(pred_inv_depth.shape)


def run_image(model, image_path, output_dir):
    img = io.imread(image_path)
    prediction = model.predict(img)

    debug_image = model.visualize(img, prediction)
    io.imsave('demo.png', debug_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, help='The image path')
    parser.add_argument('-v', '--video_path', type=str, help='The video path')
    parser.add_argument('-o', '--output_dir', type=str, help='The output dir', default="./output")
    args = parser.parse_args()

    image_path = './MegaDepth/demo.jpg'

    model = MegaDepth()
    run_image(model, image_path, args.output_dir)

