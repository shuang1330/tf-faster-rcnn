import _init_paths
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import collections
import numpy as np
from prune_with_classification_guidance import detect_diff_all
import random
from random import shuffle
from nets.vgg16 import vgg16
from tensorflow.python import pywrap_tensorflow


if __name__=='__main__':
    demonet = 'vgg16_faster_rcnn_iter_70000.ckpt'
    dataset = 'voc_2007_trainval'
    tfmodel = os.path.join('../output','vgg16',dataset, 'default', demonet)

    heatmap_path='./activations_res/res.npy'
    heatmap_all_ind = detect_diff_all(heatmap_path)

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
            net.create_architecture(sess,'TRAIN',21,tag='default',
                                    anchor_scales=[8,16,32],
                                    filter_num=[64,64,128,128,256,256,256,512,512,512,512,512,512,512])
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

    i = 0
    for key in dic:
        i += 1
        if i == 11:
            # print key
            # print dic[key]['weights']['value'].shape
            current_sum = np.sum(dic[key]['weights']['value'], axis = (0,1,2))
            # print current_sum.shape

    layer_ind = 10
    clas_index = heatmap_all_ind['%dth_acts'%(layer_ind+1)]
    random.seed(100)
    random_index = range(len(current_sum))
    shuffle(random_index)
    common_set = set(random_index[:256])&set(clas_index[:256])
    print 'there is a common set between the random pruning and the classifcation-based methods with length %d'%len(common_set)
    common_list = list(common_set)

    for common_index in common_list:
        path = '../../deep_dream/dream_results/conv5_1/%d.jpg'%(common_index)
        # if not os.path.exists(path):
            # print 'not existed the image'
        new_path = '../../deep_dream/dream_results/conv5_1/common/%d.jpg'%(common_index)
        os.rename(path,new_path)
