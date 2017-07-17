#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import _init_paths
from model.config import cfg

from datasets.factory import get_imdb
from model.test import test_net

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import collections
from random import shuffle
import random

from nets.vgg16 import vgg16
from tensorflow.python import pywrap_tensorflow
from prune_with_classification_guidance import detect_diff_all


def choose_filters(current_sum,layer_ind,heatmap_all_ind=None,random_seed = None,
                    MAGNITUTE=False,RANDOM=False,CLASSIFICATION_BASED=False,
                    COMBINED = False):
    if CLASSIFICATION_BASED:
        print 'choose pruning guided by classfiication mode'
        return heatmap_all_ind['%dth_acts'%(layer_ind+1)]
    if RANDOM:
        print 'choose random pruning mode'
        random.seed(random_seed)
        print 'using random seed %d'%random_seed
        shuffled_list = range(len(current_sum))
        shuffle(shuffled_list)
        return shuffled_list
    if MAGNITUTE:
        print 'choosing magnitute mode'
        return np.argsort(current_sum)
    if COMBINED:
        clas_index = heatmap_all_ind['%dth_acts'%(layer_ind+1)]
        random.seed(100)
        random_index = range(len(current_sum))
        shuffle(random_index)
        common_set = set(random_index[:256])&set(clas_index[:256])
        print 'there is a common set between the random pruning and the classifcation-based methods with length %d'%len(common_set)
        return list(common_set)

def filter(dic,old_filter_num,new_filter_num,heatmap_path=None,random_seed = None,
            RANDOM=False,MAGNITUTE=False,CLASSIFICATION_BASED=False,COMBINED=False):
    '''
    modify the weights_dic according the pruning rule(new_filter_num)
    inputs&output: collections.OrderedDict()
    '''
    biases = []
    weights = []
    w_momentum = []
    b_momentum = []
    name_scopes = []
    for name_scope in dic:
        name_scopes.append(name_scope)
        for name in dic[name_scope]:
            if name.startswith('weights'):
                # weights.append(dic[name_scope][name])
                weights.append(dic[name_scope][name]['value'])
                w_momentum.append(dic[name_scope][name]['Momentum'])
            elif name.startswith('biases'):
                # biases.append(dic[name_scope][name])
                biases.append(dic[name_scope][name]['value'])
                b_momentum.append(dic[name_scope][name]['Momentum'])

    diff = [(old_filter_num[ind] - new_filter_num[ind])
            for ind in range(len(old_filter_num))] #
    heatmap_all_ind={}
    if CLASSIFICATION_BASED or COMBINED:
        heatmap_all_ind = detect_diff_all(heatmap_path)
    current_ind = 0
    pre_ind = 0
    if diff[0] != 0: # for the oth layer
        current_sum = np.sum(weights[0], axis = (0,1,2))
        current_ind = choose_filters(current_sum,0,heatmap_all_ind,
                                    random_seed,
                                    RANDOM=RANDOM,MAGNITUTE=MAGNITUTE,
                                    CLASSIFICATION_BASED=CLASSIFICATION_BASED,
                                    COMBINED=COMBINED)
        weights[0] = np.delete(weights[0], current_ind[:diff[0]], axis = 3)
        biases[0] = np.delete(biases[0],current_ind[:diff[0]], axis = 0)
        if w_momentum[0] is not None:
            w_momentum[0] = np.delete(w_momentum[0],
                                        current_ind[:diff[0]],
                                        axis = 3)
            b_momentum[ind] = np.delete(b_momentum[ind],
                                        current_ind[:diff[0]],
                                        axis = 0)

    pre_ind = current_ind
    current_ind = None
    for ind in range(1,len(old_filter_num)): # for every layer
        if diff[ind-1] != 0:
            weights[ind] = np.delete(weights[ind], pre_ind[:diff[ind-1]],
                                    axis = 2)
            if diff[ind] == 0:
                pre_ind = None
        if diff[ind] != 0:
            current_sum = np.sum(weights[ind],axis = (0,1,2))
            current_ind = choose_filters(current_sum,ind,heatmap_all_ind,
                                        random_seed,
                                        RANDOM=RANDOM,MAGNITUTE=MAGNITUTE,
                                        CLASSIFICATION_BASED=CLASSIFICATION_BASED,
                                        COMBINED=COMBINED)
            # print current_ind
            weights[ind] = np.delete(weights[ind], current_ind[:diff[ind]],
                                    axis = 3)
            biases[ind] = np.delete(biases[ind],current_ind[:diff[ind]],
                                    axis = 0 )
            if w_momentum[ind] is not None:
                w_momentum[ind] = np.delete(w_momentum[ind],
                                            current_ind[:diff[ind]],
                                            axis = 3)
                b_momentum[ind] = np.delete(b_momentum[ind],
                                            current_ind[:diff[ind]],
                                            axis = 0)
            pre_ind = current_ind
            current_ind = None

    ind = 0
    # concatenate
    for name_scope in dic:
        for name in dic[name_scope]:
            if ind <= len(old_filter_num):
                if name.startswith('weights'):
                    dic[name_scope][name]['value'] = weights[ind]
                    dic[name_scope][name]['Momentum'] = w_momentum[ind]
                elif name.startswith('biases'):
                    dic[name_scope][name]['value'] = biases[ind]
                    dic[name_scope][name]['Momentum'] = b_momentum[ind]
        ind += 1
    return dic

def prune_net_for_training(tfmodel,weights_path,old_filter_num,new_filter_num,
                            heatmap_path=None,random_seed = None,
                            SAVE=False,RANDOM=False,MAGNITUTE=False,
                            CLASSIFICATION_BASED=False,COMBINED=False):
    '''
    prune weights in ckpt files
    save results in npy files
    '''
    reader = pywrap_tensorflow.NewCheckpointReader(tfmodel)
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # load the network
    name_scopes = []
    with tf.Graph().as_default() as g1:
        with tf.Session(config=tfconfig, graph=g1).as_default() as sess:
            #load network
            net = vgg16(batch_size=1)
            filter_num = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]
            net.create_architecture(sess,'TRAIN',21,tag='default',
                                    anchor_scales=[8,16,32],
                                    filter_num=filter_num)
            # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            #     print(var.name,var.get_shape())
            saver = tf.train.Saver()
            saver.restore(sess,tfmodel)
            print 'Loaded network {:s}'.format(tfmodel)
            #get the weights from ckpt file
            dic = collections.OrderedDict()
            all_variables = []
            for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                name_list = item.name[:-2].split('/')
                name_scopes.append('/'.join(name_list[:-1]))

            for name_scope in name_scopes:
                for name in ('weights','biases'):#
                    tensor_name = '{}/{}'.format(name_scope,name)
                    momentum_name = '{}/Momentum'.format(tensor_name)
                    if name_scope not in dic:
                        try:
                            dic[name_scope] = \
                            {name:{'value':reader.get_tensor(tensor_name),
                            'Momentum':reader.get_tensor(momentum_name)}}
                        except:
                            dic[name_scope] = \
                            {name:{'value':reader.get_tensor(tensor_name),
                            'Momentum':None}}
                    else:
                        dic[name_scope][name] = \
                        {'value':reader.get_tensor(tensor_name),
                        'Momentum':None}
                        try:
                            dic[name_scope][name]['Momentum'] = \
                            reader.get_tensor(momentum_name)
                        except:
                            continue

    # filter the weights
    print 'using random seed %d'%random_seed
    dic = filter(dic,old_filter_num,new_filter_num,heatmap_path,random_seed,
                 RANDOM,MAGNITUTE,CLASSIFICATION_BASED,COMBINED)
    # raise NotImplementedError

    # display the filtered weights' shape
    # for name_scope in dic:
    #     for name in dic[name_scope]:
    #         try:
    #             print 'After filtering, the variable {}/{} has shape {}'.format(
    #             name_scope, name, dic[name_scope][name]['Momentum'].shape)
    #         except:
    #             print 'After filtering, the variable {}/{}/Momentum is: '.format(
    #             name_scope, name, dic[name_scope][name]['Momentum'])

    # save the pruned weights to npy file
    if SAVE:
        np.save(weights_path, dic)
        print 'The weights are saved in {}'.format(weights_path)
    return dic

if __name__ == '__main__':
    # Below are variables that needs checking every time
    num_classes = 21
    old_filter_num = (64,64,128,128,256,256,256,512,512,512,512,512,512,512)
    new_filter_num = (64,64,64,128,256,256,256,512,512,512,512,512,512,512)
    names = ('weights','biases')
    # Defined path for loading original weights from ckpy files
    demonet = 'vgg16_faster_rcnn_iter_70000.ckpt'
    dataset = 'voc_2007_trainval'
    tfmodel = os.path.join('../output','vgg16',dataset, 'default', demonet)
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError('{:s} not found'.format(tfmodel + '.meta'))
    # Defined path for saving pruned weights to npy files
    weights_name = 'combined_prune_conv3_to64_with_momentum.npy'
    folder_path = '../output/pruning/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    weights_path = os.path.join(folder_path,weights_name)
    # Only useful when classification-based method is chosen
    heatmap_path = './activations_res/res.npy'
    heatmap_all_ind = {}
    # seed = 100
    # weights_dic = prune_net_for_training(tfmodel,
    #                             weights_path,
    #                             old_filter_num,
    #                             new_filter_num,
    #                             heatmap_path='./activations_res/res.npy',
    #                             random_seed = seed,
    #                             SAVE=False,RANDOM=True,MAGNITUTE=False,
    #                             CLASSIFICATION_BASED=False,
    #                             COMBINED=False)
    #
    #
    # '''
    # load the new weights to a new graph,
    # test the pruned network
    # '''
    # # set config
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth=True
    # with tf.Graph().as_default() as g2:
    #     with tf.Session(config=tfconfig,graph=g2).as_default() as sess:
    #         #load the new graph
    #         net = vgg16(batch_size=1)
    #         net.create_architecture(sess,'TEST',num_classes,tag='default',
    #                                 anchor_scales = [8,16,32],
    #                                 filter_num = new_filter_num)
    #
    #         # load the new weights from npy file
    #         # weights_dic = np.load(weights_path).item()
    #
    #         for name_scope in weights_dic:
    #             with tf.variable_scope(name_scope,reuse = True):
    #                 for name in weights_dic[name_scope]:
    #                     var = tf.get_variable(name)
    #                     sess.run(var.assign(weights_dic[name_scope][name]['value']))
    #         print 'assigned pruned weights to the pruned model'
    #
    #         # test the new model
    #         imdb = get_imdb('voc_2007_test')
    #         filename = 'demo_pruning/experiments/random/random_seed%d'%seed
    #         experiment_setup = 'prune_layer3_to64'
    #         test_net(sess, net, imdb, filename,
    #                 experiment_setup=experiment_setup,
    #                 max_per_image=100)

    seeds = [200,300,400,500,600,700,800] # random seed for random method
    for seed in seeds:
        weights_dic = prune_net_for_training(tfmodel,
                                    weights_path,
                                    old_filter_num,
                                    new_filter_num,
                                    heatmap_path='./activations_res/res.npy',
                                    random_seed = seed,
                                    SAVE=False,RANDOM=True,MAGNITUTE=False,
                                    CLASSIFICATION_BASED=False,
                                    COMBINED=False)


        '''
        load the new weights to a new graph,
        test the pruned network
        '''
        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True
        with tf.Graph().as_default() as g2:
            with tf.Session(config=tfconfig,graph=g2).as_default() as sess:
                #load the new graph
                net = vgg16(batch_size=1)
                net.create_architecture(sess,'TEST',num_classes,tag='default',
                                        anchor_scales = [8,16,32],
                                        filter_num = new_filter_num)

                # load the new weights from npy file
                # weights_dic = np.load(weights_path).item()

                for name_scope in weights_dic:
                    with tf.variable_scope(name_scope,reuse = True):
                        for name in weights_dic[name_scope]:
                            var = tf.get_variable(name)
                            sess.run(var.assign(weights_dic[name_scope][name]['value']))
                print 'assigned pruned weights to the pruned model'

                # test the new model
                imdb = get_imdb('voc_2007_test')
                filename = 'demo_pruning/experiments/random/random_seed_variation'
                experiment_setup = 'prune_layer3_to64_seed%d'%seed
                test_net(sess, net, imdb, filename,
                        experiment_setup=experiment_setup,
                        max_per_image=100)
