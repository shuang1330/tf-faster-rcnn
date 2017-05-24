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

import tensorflow as tf
from nets.vgg16 import vgg16

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

if __name__ == '__main__':
  cfg_file = '../experiments/cfgs/vgg16.yml'
  imdb_name = 'voc_2007_test'
  net = 'vgg16'
  comp_mode=False
  max_per_image=300
  model = \
  '../output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt'
  set_cfgs=['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
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
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS,
                          filter_num = (64,64,128,128,256,256,256,\
                          512,512,512,512,512,512,512))

  print(('Loading model check point from {:s}').format(model))
  saver = tf.train.Saver()
  saver.restore(sess, model)
  print('Loaded.')

  im_names = ['000456.jpg']
  im_file = os.path.join(cfg.DATA_DIR, 'demo', im_names[0])
  im = cv2.imread(im_file)
  scores, boxes, _ = im_detect(sess, net, im)
  print(np.array(scores).shape)
  print(np.array(boxes).shape)

  # test_net(sess, net, imdb, filename, max_per_image=max_per_image)

  sess.close()
