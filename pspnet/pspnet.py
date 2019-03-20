import sys
sys.path.insert(0, './PSPNet-Keras-tensorflow')

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from keras import backend as K

from model.pspnet import get_pspnet
import utils

class PSPNet:
    def __init__(self):
    	self.flip = False
    	self.sess = tf.Session()
    	K.set_session(self.sess)
    	with self.sess.as_default():
    		self.model = get_pspnet("pspnet50_ade20k", None)

    def predict(self, img):
    	with self.sess.as_default():
	        probs = self.model.predict(img, self.flip)
	        cm = np.argmax(probs, axis=2)
	        return cm

    def visualize(self, img, prediction):
        color_cm = utils.add_color(prediction)
        alpha_blended = 0.5 * color_cm * 255 + 0.5 * img
        return alpha_blended


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
            continue

        prediction = model.predict(img)
        cache.append((img, prediction))

    print("Visualize")
    # Prepare output video
    out_fn = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_fn, fourcc, fps, (frame_width, frame_height))
    for img, prediction in tqdm(cache):
        debug_image = model.visualize(img, prediction)
        
        if debug_image.ndim == 2:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
            debug_image = cv2.resize(debug_image, (frame_width, frame_height))
        out.write(debug_image)

    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, help='The image path')
    parser.add_argument('-v', '--video_path', type=str, help='The video path')
    parser.add_argument('-o', '--output_dir', type=str, help='The output dir', default="./output")
    args = parser.parse_args()

    model = PSPNet()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.image_path:
        run_image(model, args.image_path, args.output_dir)
    elif args.video_path:
        run_video(model, args.video_path, args.output_dir)
    else:
        print("Choose image or video.")
