'''
# This code is based on DrSleep's framework and hellochick's code
  DrSleep: https://github.com/DrSleep/tensorflow-deeplab-resnet 
  hellochick: https://github.com/hellochick/PSPNet-tensorflow

# Author: Shaochun Ning 

# Command:
  
  python3 evaluate.py 

  python3 evaluate.py --flipped-eval 

'''

from __future__ import print_function
import argparse
import os
import sys
import time

from PIL import Image
import tensorflow as tf
import numpy as np
from scipy import misc

from tqdm import trange
from model import PSPNet101, PSPNet50
from tools import *
import time

starttime = time.clock()

save_dir='./dataset/ground_truth_evaluate/'   

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
               
my_own_param = {'crop_size': [720, 720],
                    'num_classes': 18,
                    'ignore_label': 0,
                    'num_steps': 100,
                    'model': PSPNet101,
                    'data_dir': './dataset/train/', 
                    'val_list': './list_own/list_val'}     

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='my_own')

    return parser.parse_args()

def main():
    args = get_arguments()

    # load parameters

    param = my_own_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    ignore_label = param['ignore_label']
    num_steps = param['num_steps']
    PSPNet = param['model']
    data_dir = param['data_dir']

    # Set placeholder 
    image_filename = tf.placeholder(dtype=tf.string)
    anno_filename = tf.placeholder(dtype=tf.string)

    # Read & Decode image
    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
    img.set_shape([None, None, 3])
    anno.set_shape([None, None, 1])

    shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], shape[0]), tf.maximum(crop_size[1], shape[1]))
    img = preprocess(img, h, w)

     # Create network.
    net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(tf.squeeze(img))
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

    raw_output = net.layers['conv6']

    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Scale feature map to image size, get prediction
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[0], shape[1])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    pred_orig = decode_labels(raw_output_up, shape, num_classes)

    # Calculate mIoU
    pred_flatten = tf.reshape(pred, [-1,])
    raw_gt = tf.reshape(anno, [-1,])
    indices = tf.squeeze(tf.where(tf.not_equal(raw_gt, ignore_label)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    ckpt_path='./model_own/model.ckpt-100'
    loader = tf.train.Saver(var_list=tf.global_variables())
    loader.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


    file = open(param['val_list'], 'r') 
    for step in trange(num_steps, desc='evaluation', leave=True):
        f1, f2 = file.readline().split('\n')[0].split(' ')
        f11 = os.path.join(data_dir, f1)
        f22 = os.path.join(data_dir, f2)

        preds = sess.run(pred_orig,feed_dict={image_filename: f11})  
        _ = sess.run(update_op, feed_dict={image_filename: f11, anno_filename: f22})
        
        misc.imsave(save_dir + f1.split('/')[-1], preds[0])
    print('mIoU: {:04f}'.format(sess.run(mIoU)))
    

    endtime = time.clock()
    print(endtime - starttime)
   
if __name__ == '__main__':
    main()
