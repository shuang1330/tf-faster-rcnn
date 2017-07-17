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
from math import ceil

from tensorflow.python import pywrap_tensorflow
from prune_with_classification_guidance import detect_diff_all
from prune import prune_net_for_training

if __name__=='__main__':
    # Defined path for loading original weights from ckpy files
    demonet = 'vgg16_faster_rcnn_iter_70000.ckpt'
    dataset = 'voc_2007_trainval'
    tfmodel = os.path.join('../output','vgg16',dataset, 'default', demonet)
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError('{:s} not found'.format(tfmodel + '.meta'))
    # define pruning strategy
    num_classes = 21
    old_filter_num = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]
    new_filter_num = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]
    names = ('weights','biases')
    # Defined path for saving pruned weights to npy files
    folder_path = '../output/pruning/'
    # Initialize heamap path for classification-based methods
    heatmap_path = None

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for index_of_layer in range(10,11): # different layer
        new_filter_num = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]
        for num_of_filter in [0.5]: # different filters for each layer
            new_filter_num[index_of_layer] = int(new_filter_num[index_of_layer]*num_of_filter)
            print 'The new graph is: ',new_filter_num
            # load the new grph
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth=True
            with tf.Graph().as_default() as g2:
                with tf.Session(config=tfconfig,graph=g2).as_default() as sess:
                    net = vgg16(batch_size=1)
                    net.create_architecture(sess,'TEST',num_classes,tag='default',
                                            anchor_scales = [8,16,32],
                                            filter_num = new_filter_num)
                    saver = tf.train.Saver()
                    # different pruning method
                    for method in ['classification-based']:
                        weights_name = '%s_prune_conv%d_to%d_with_momentum.ckpt'%(
                            method,index_of_layer,new_filter_num[index_of_layer])
                        weights_path = os.path.join(folder_path,weights_name)
                        if method=='random':
                            RANDOM,MAGNITUTE,CLASSIFICATION_BASED=True,False,False
                        elif method=='magnitute':
                            RANDOM,MAGNITUTE,CLASSIFICATION_BASED=False,True,False
                        elif method=='classification-based':
                            RANDOM,MAGNITUTE,CLASSIFICATION_BASED=False,False,True
                            heatmap_path = './activations_res/res.npy'
                        else:
                            raise IOError('No method is chosen')
                        weights_dic=prune_net_for_training(tfmodel,
                                                weights_path,
                                                old_filter_num,
                                                new_filter_num,
                                                heatmap_path=heatmap_path,
                                                SAVE=False,RANDOM=RANDOM,
                                                MAGNITUTE=MAGNITUTE,
                                                CLASSIFICATION_BASED=CLASSIFICATION_BASED)

                        # load the new weights from npy file
                        # weights_dic = np.load(weights_path).item()
                        for name_scope in weights_dic:
                            with tf.variable_scope(name_scope,reuse = True):
                                for name in weights_dic[name_scope]:
                                    var = tf.get_variable(name)
                                    sess.run(var.assign(weights_dic[name_scope][name]['value']))
                        print 'assigned pruned weights to the pruned model'
                        if True:
                            saver.save(sess,weights_path)
                        raise NotImplementedError
                        # test the new model
                        imdb = get_imdb('voc_2007_test')
                        filename = 'demo_pruning/experiments2'
                        experiment_setup = '%s/random_seed100/layer%d/to%d'%(method,
                                                index_of_layer,
                                                new_filter_num[index_of_layer])
                        test_net(sess, net, imdb, filename,
                                experiment_setup=experiment_setup,
                                max_per_image=100)
