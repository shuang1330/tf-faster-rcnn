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
import copy

from nets.vgg16 import vgg16
from tensorflow.python import pywrap_tensorflow

def delete_filters(variables,current_ind,diff):
    print diff
    print variables[0].shape
    variables[0] = np.delete(variables[0],current_ind[:diff],axis=3)
    variables[1] = np.delete(variables[1],urrent_ind[:diff],axis=0)
    return variables

def list_normalizer(ori_list):
    max_val = ori_list.max()
    min_val = ori_list.min()
    if max_val == 0:
        return ori_list
    normalized_list = [(i-min_val)/(max_val-min_val) for i in ori_list]
    return normalized_list

class PruningWrapper():
    def __init__(self,num_classes,layer_2prune,pruned_filter_num,
                old_filter_num,SAVE,MOMENTUM,METHOD):
        # defineing graph
        self.num_classes = num_classes
        self.old_filter_num =old_filter_num
        self.new_filter_num =old_filter_num
        self.variable_type = ['weights','biases']

        # defining path for original weights
        self.old_model_path = os.path.join('..','output','vgg16',
                                            'voc_2007_trainval',
                                            'default',
                                            'vgg16_faster_rcnn_iter_70000.ckpt')
        if not os.path.isfile(self.old_model_path + '.meta'):
            raise IOError('{:s} not found'.format(self.old_model_path + '.meta'))

        # define pruing methods
        self.METHOD = METHOD
        self.layer_2prune = layer_2prune
        self.pruned_filter_num = pruned_filter_num

        # define path for saving pruned weights
        self.SAVE = SAVE
        self.MOMENTUM = MOMENTUM
        self.weights_path = None

        # define activations files for classification-based method
        self.heatmap_path = os.path.join('..','tools','activations_res','res.npy')
        self.heatmap_all_ind = {}

        # define the subset of classes to prune
        self.CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
        self.class_to_ind = dict(list(zip(self.CLASSES,
                                list(range(len(self.CLASSES))))))
        self.SUB_CLAS = ('bicycle', 'bus', 'car','motorbike', 'person', 'train')

    def detect_diff_one_layer(self,norm_hm_one_layer):
        interest_average = np.zeros((norm_hm_one_layer.shape[1],))
        diff_ind = np.zeros((norm_hm_one_layer.shape[1],))
        amplifier = 10
        for clas in self.SUB_CLAS:
            ind = self.class_to_ind[clas]
            interest_average[:] += norm_hm_one_layer[ind]
        interest_average = interest_average/len(self.SUB_CLAS)
        for clas in self.CLASSES:
            if clas not in self.SUB_CLAS:
                ind = self.class_to_ind[clas]
                temp = amplifier*(norm_hm_one_layer[ind]-interest_average)
                diff_ind += temp
        diff_ind = np.argsort(diff_ind)[::-1]
        return diff_ind

    def detect_diff_all(self):
        hm_all = np.load(self.heatmap_path).item()
        norm_hm_all = {}
        hm_ind = {} # dictionary to record the diff_ind for every layer
        sub_clas_index = [self.class_to_ind[i] for i in self.SUB_CLAS]
        for key in hm_all: # for evey layer
            norm_hm_all[key] = np.zeros(hm_all[key].shape,np.float32)
            for i,sub_list in enumerate(hm_all[key]): # for every row in the layer
                norm_hm_all[key][i,:] = list_normalizer(sub_list)
            hm_ind[key] = self.detect_diff_one_layer(norm_hm_all[key]) # [21, 64/...]
        return hm_ind

    def get_new_filter_num(self):
        self.new_filter_num = copy.copy(self.old_filter_num)
        self.new_filter_num[self.layer_2prune-1] = self.pruned_filter_num
        return self.new_filter_num

    def choose_filters(self,current_sum,layer_ind):
        if self.METHOD=='CLASSIFICATION_BASED':
            print 'choose pruning guided by classfiication mode'
            self.heatmap_all_ind = self.detect_diff_all()
            return self.heatmap_all_ind['%dth_acts'%(layer_ind+1)]
        if self.METHOD=='RANDOM':
            print 'choose random pruning mode'
            shuffled_list = range(len(current_sum))
            shuffle(shuffled_list)
            return shuffled_list
        if self.METHOD=='MAGNITUTE':
            print 'choosing magnitute mode'
            return np.argsort(current_sum)

    def load_GraphVars_from_reader(self,variables_list):
        reader = pywrap_tensorflow.NewCheckpointReader(self.old_model_path)
        weights_dic = collections.OrderedDict()
        name_scopes = []
        for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            name_list = item.name[:-2].split('/')
            name_scopes.append('/'.join(name_list[:-1]))

        for name_scope in name_scopes:
            for name in self.variable_type[:-1]:
                tensor_name = '{}/{}'.format(name_scope,name)
                momentum_name = '{}/Momentum'.format(tensor_name)
                if name_scope not in weights_dic:
                    try:
                        weights_dic[name_scope] = \
                        {name:{'value':reader.get_tensor(tensor_name),
                        'Momentum':reader.get_tensor(momentum_name)}}
                    except:
                        weights_dic[name_scope] = \
                        {name:{'value':reader.get_tensor(tensor_name),
                        'Momentum':None}}
                else:
                    weights_dic[name_scope][name] = \
                    {'value':reader.get_tensor(tensor_name),
                    'Momentum':None}
                    if self.MOMENTUM: # record momentum if training is required
                        try:
                            weights_dic[name_scope][name]['Momentum'] = \
                            reader.get_tensor(momentum_name)
                        except:
                            continue
        return weights_dic

    def filter(self,weights_dic):
        biases = []
        weights = []
        w_momentum = []
        b_momentum = []
        name_scopes = []
        for key in weights_dic:
            for subkey in weights_dic[key]:
                print key,subkey
        for name_scope in weights_dic:
            name_scopes.append(name_scope)
            for name in weights_dic[name_scope]:
                if name.startswith('weights'):
                    weights.append(weights_dic[name_scope][name]['value'])
                    print weights_dic[name_scope][name]['value'].shape
                    w_momentum.append(weights_dic[name_scope][name]['Momentum'])
                elif name.startswith('biases'):
                    biases.append(weights_dic[name_scope][name]['value'])
                    b_momentum.append(weights_dic[name_scope][name]['Momentum'])
        print '\n\n\nthe weights list is supposed to be of shape: ', len(weights)
        print '\n\n\nthe biases list is supposed the same: ', len(biases)
        raise NotImplementedError
        self.new_filter_num = self.get_new_filter_num()
        diff = [(self.old_filter_num[ind] - self.new_filter_num[ind])
                for ind in range(len(self.old_filter_num))]

        current_ind = 0
        pre_ind = 0
        if diff[0] != 0: # for the 0th layer
            current_sum = np.sum(weights[0], axis = (0,1,2))
            current_ind = self.choose_filters(current_sum,0)
            [weights[0],biases[0]] = delete_filters([weights[0],biases[0]],
                                                    current_ind,diff[0])

            if w_momentum[0] is not None:
                [w_momentum[0],b_momentum[0]] = delete_filters([w_momentum[0],
                                                b_momentum[0]],current_ind,diff[0])

        pre_ind = current_ind
        current_ind = None
        for ind in range(1,len(old_filter_num)): # for every layer
            if diff[ind-1] != 0:
                weights[ind] = np.delete(weights[ind],
                                        pre_ind[:diff[ind-1]],
                                        axis = 2)
                if diff[ind] == 0:
                    pre_ind = None
            if diff[ind] != 0:
                current_sum = np.sum(weights[ind],axis = (0,1,2))
                current_ind = self.choose_filters(current_sum,ind)
                print 'current layer to prune', current_ind
                print len(weights)
                print len(biases)
                [weights[ind],biases[ind]] = delete_filters([weights[ind],
                                                        biases[ind]],
                                                        current_ind,diff[ind])
                if w_momentum[ind] is not None:
                    [w_momentum[ind],b_momentum[ind]] = delete_filters([w_momentum[ind],
                                                    b_momentum[ind]],current_ind,diff[ind])

                pre_ind = current_ind
                current_ind = None

        ind = 0
        # load pruned weights to a dictionary and return
        for name_scope in weights_dic:
            for name in weights_dic[name_scope]:
                if ind <= len(old_filter_num):
                    if name.startswith('weights'):
                        weights_dic[name_scope][name]['value'] = weights[ind]
                        weights_dic[name_scope][name]['Momentum'] = w_momentum[ind]
                    elif name.startswith('biases'):
                        weights_dic[name_scope][name]['value'] = biases[ind]
                        weights_dic[name_scope][name]['Momentum'] = b_momentum[ind]
            ind += 1
        return weights_dic

    def prune_net_for_training(self):
        # load the network
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True
        with tf.Graph().as_default() as g1:
            with tf.Session(config=tfconfig, graph=g1).as_default() as sess:
                #load network
                net = vgg16(batch_size=1)
                net.create_architecture(sess,'TRAIN',self.num_classes,
                                        tag='default',
                                        anchor_scales=[8,16,32],
                                        filter_num=self.old_filter_num)
                saver = tf.train.Saver()
                saver.restore(sess,self.old_model_path)
                print 'Loaded network {:s}'.format(self.old_model_path)
                # get the weights'names from ckpt file
                weights_dic = self.load_GraphVars_from_reader(
                               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        # filter the weights
        dic = self.filter(weights_dic)
        for key in dic:
            for subkey in dic[key]:
                print dic[key][subkey]['value'].shape

        if self.SAVE:
            weights_name = '%s_layer%d_to%d%'%(self.METHOD,self.layer_2prune,self.pruned_filter_num)
            folder_path = os.path.join('..','output','pruning')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            self.weights_path = os.path.join(folder_path,weights_name)
            np.save(weights_path, weights_dic)
            print 'The weights are saved in {}'.format(weights_path)
        return dic

if __name__ == '__main__':
    old_filter_num = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]
    prunesolver = PruningWrapper(num_classes=21,layer_2prune=11,
                                pruned_filter_num=128,
                                old_filter_num=old_filter_num,
                                SAVE=False,MOMENTUM=True,
                                METHOD='CLASSIFICATION_BASED')
    weights_dic = prunesolver.prune_net_for_training()

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
            net.create_architecture(sess,'TEST',21,tag='default',
                                    anchor_scales = [8,16,32],
                                    filter_num = prunesolver.new_filter_num)
            print(prunesolver.new_filter_num)

            for name_scope in weights_dic:
                with tf.variable_scope(name_scope,reuse = True):
                    for name in weights_dic[name_scope]:
                        var = tf.get_variable(name)
                        # print weights_dic[name_scope][name]['value'].shape
                        # sess.run(var.assign(weights_dic[name_scope][name]['value']))
            raise NotImplementedError
            print 'assigned pruned weights to the pruned model'


            # test the new model
            imdb = get_imdb('voc_2007_test')
            filename = 'demo_pruning/experiments/random'
            experiment_setup = 'prune_layer%d_to%d'%(prunesolver.layer_2prune,
                                prunesolver.pruned_filter_num)
            test_net(sess, net, imdb, filename,
                    experiment_setup=experiment_setup,
                    max_per_image=100)
