import sys
sys.path.insert(0, './MegaDepth')

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model

class MegaDepth:

    def __init__(self):
        # Fix symlink checkpoint dir
        self.setup()

    def setup(self):
        self.model = create_model(opt)
        self.model.switch_to_eval()

    def preprocess(self, img):
        input_height = 384
        input_width  = 512
        img = img[:,:,::-1]
        img = np.float32(img)/255.0
        img = cv2.resize(img, (input_width, input_height))
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
        debug_image = 255 * pred_inv_depth
        return debug_image


def run_image(model, image_path, output_dir):
    img = cv2.imread(image_path)
    prediction = model.predict(img)
    debug_image = model.visualize(img, prediction)

    out_fn = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_fn, debug_image)
    print("Wrote to", out_fn)

def run_video(model, video_path, output_dir):
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

        prediction = model.predict(img)
        cache.append((img, prediction))

    print("Visualize")
    # Prepare output video
    out_fn = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_fn, fourcc, fps, (frame_width, frame_height))
    for i in tqdm(range(len(cache))):
        img, prediction = cache[i]
        debug_image = model.visualize(img, prediction)
        out.write(debug_image)

    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, help='The image path')
    parser.add_argument('-v', '--video_path', type=str, help='The video path')
    parser.add_argument('-o', '--output_dir', type=str, help='The output dir', default="./output")
    args = parser.parse_args()

    model = MegaDepth()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.image_path:
        run_image(model, args.image_path, args.output_dir)
    elif args.video_path:
        run_video(model, args.video_path, args.output_dir)
    else:
        print("Choose image or video.")


