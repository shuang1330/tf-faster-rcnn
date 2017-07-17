#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import _init_paths
from model.config import cfg

from datasets.factory import get_imdb
from model.test import test_net

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import collections

from nets.vgg16 import vgg16
from tensorflow.python import pywrap_tensorflow

num_classes = 21
old_filter_num = (64,64,128,128,256,256,256,512,512,512,512,512,512,512)
new_filter_num = (64,64,128,128,256,256,256,512,512,512,128,512,512,512)
names = ('weights','biases')
# Defined path for saving pruned weights to npy files
weights_name = 'pruned_conv11_momentum.npy'
folder_path = '../output/pruning/'
weights_path = os.path.join(folder_path,weights_name)

if __name__=='__main__':
    '''
    load the new weigts to a new graph,
    test the pruned network
    '''
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    with tf.Graph().as_default() as g:
        with tf.Session(config=tfconfig,graph=g).as_default() as sess:
            #load the new graph
            net = vgg16(batch_size=1)
            net.create_architecture(sess,'TEST',num_classes,tag='default',
                                    anchor_scales = [8,16,32],
                                    filter_num = new_filter_num)

            # load the new weights from npy file
            weights_dic = np.load(weights_path).item()

            for name_scope in weights_dic:
                with tf.variable_scope(name_scope,reuse = True):
                    for name in weights_dic[name_scope]:
                        var = tf.get_variable(name)
                        sess.run(var.assign(weights_dic[name_scope][name]['value']))
                        print 'assign pretrain model to {}/{}'.format(name_scope,name)

            # test the new model
            imdb = get_imdb('voc_2007_test')
            filename = 'demo_pruning'
            test_net(sess, net, imdb, filename, max_per_image=100)
