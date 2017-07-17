from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from model.trainval_pruned import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import pprint
import numpy as np
import sys

import tensorflow as tf
from nets.vgg16 import vgg16

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb


if __name__ == '__main__':

  cfg_file = '../experiments/cfgs/vgg16.yml'
  set_cfgs = None
  imdb_name = 'voc_2007_trainval'
  imdbval_name = 'voc_2007_test'
  tag = None
  net = 'vgg16'
  output_dir = '../output/retraining'
  tb_dir = '/volume/home/shuang/tf-faster-rcnn/tensorboard/vgg16/voc_2007_trainval/pruned'
  weight = '../output/pruning/random_prune_conv10_to256_with_momentum.npy'

  if cfg_file is not None:
    cfg_from_file(cfg_file)
  if set_cfgs is not None:
    cfg_from_list(set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)

  # train set
  imdb, roidb = combined_roidb(imdb_name)
  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
#  output_dir = get_output_dir(imdb, tag)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, tag)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  if net == 'vgg16':
    net = vgg16(batch_size=cfg.TRAIN.IMS_PER_BATCH)
  else:
    raise NotImplementedError
  filter_num = [64,64,128,128,256,256,256,512,512,512,256,512,512,512]
  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,filter_num,
            pretrained_model=weight,
            max_iters=5000)
