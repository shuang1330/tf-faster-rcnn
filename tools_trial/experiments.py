#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import _init_paths
from datasets.factory import get_imdb
from model.test import test_net
from nets.vgg16 import vgg16

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import collections
from random import shuffle
from math import ceil

from tensorflow.python import pywrap_tensorflow
from prune_with_classification_guidance import detect_diff_all
from prune import PruningWrapper

if __name__=='__main__':
    # define pruning methods
    old_filter_num = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]
    num_classes = 21
    SAVE=False
    MOMENTUM=False
    for index_of_layer in [9,10]: # different layer
        for num_of_filter in [64,128,256]: # different filters for each layer
            for method in ['CLASSIFICATION_BASED','RANDOM','MAGNITUTE']:
                # prune weights
                prunesolver = PruningWrapper(num_classes=num_classes,
                                            layer2_prune=index_of_layer,
                                            pruned_filter_num=num_of_filter,
                                            old_filter_num=old_filter_num,
                                            SAVE=SAVE,
                                            MOMENTUM=MOMENTUM,
                                            METHOD=method)
                weights_dic = prunesolver.prune_net_for_training()
                # load the new graph
                tfconfig = tf.ConfigProto(allow_soft_placement=True)
                tfconfig.gpu_options.allow_growth=True
                with tf.Graph().as_default() as g2:
                    with tf.Session(config=tfconfig,graph=g2).as_default() as sess:
                        net = vgg16(batch_size=1)
                        net.create_architecture(sess,'TEST',num_classes,
                                        tag='default',
                                        anchor_scales = [8,16,32],
                                        filter_num = prunesolver.new_filter_num)
                        # load the weights
                        for name_scope in weights_dic:
                            with tf.variable_scope(name_scope,reuse = True):
                                for name in weights_dic[name_scope]:
                                    var = tf.get_variable(name)
                                    sess.run(var.assign(
                                        weights_dic[name_scope][name]['value']))
                        print 'assigned pruned weights to the pruned model'
                        # test the new model
                        imdb = get_imdb('voc_2007_test')
                        filename = 'demo_pruning/experiments/random'
                        experiment_setup = 'prune_layer%d_to%d'%(
                                            prunesolver.layer_2prune,
                                            prunesolver.pruned_filter_num)
                        test_net(sess, net, imdb, filename,
                                experiment_setup=experiment_setup,
                                max_per_image=100)
