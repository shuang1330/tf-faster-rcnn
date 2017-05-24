# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import os, cv2
from model.test import im_detect
import numpy as np
from utils.timer import Timer
from model.nms_wrapper import nms
import matplotlib.pyplot as plt

import tensorflow as tf
from nets.vgg16 import vgg16


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, _ = im_detect(sess, net, im)
    print(np.array(scores).shape)
    print(CLASSES)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

if __name__ == '__main__':
    cfg_file = '../experiments/cfgs/vgg16.yml'
    imdb_name = 'voc_2007_test'
    # imdb_name = 'coco_2014_minival'
    net = 'vgg16'
    comp_mode=False
    max_per_image=300
    # model = \
    # '../output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt'
    model = \
    '../../pretrained_weights/coco_2014_train+coco_2014_valminusminival/vgg16_faster_rcnn_iter_1190000.ckpt'
    set_cfgs={'ANCHOR_SCALES':[8,16,32], 'ANCHOR_RATIOS':[0.5,1,2]}
    tag = ''

    cfg_from_file(cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    filename = os.path.splitext(os.path.basename(model))[0]

    tag = tag if tag else 'default'
    filename = tag + '/' + filename

    imdb = get_imdb(imdb_name)
    imdb.competition_mode(comp_mode)##############################
    print('imdb.competiton_mode: {}'.format(imdb.competition_mode))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    net = vgg16(batch_size=1)
    # load model
    net.create_architecture(sess, "TEST", imdb.num_classes, tag='default',
                            anchor_scales=set_cfgs[ANCHOR_SCALES],
                            anchor_ratios=set_cfgs[ANCHOR_RATIOS])

    print(('Loading model check point from {:s}').format(model))
    saver = tf.train.Saver()
    saver.restore(sess, model)
    print('Loaded.')

    im_names = ['frame-080.jpg']#, '000542.jpg', '001150.jpg',
                #'001763.jpg', '004545.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()
