#--------------------------------------------------------
# BN: no
# RELU: yes
# l2 regularization: yes
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg

def vgg_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY):
#                     batch_norm_decay=0.997,
#                     batch_norm_epsilon=1e-5,
#                     batch_norm_scale=True):
#  batch_norm_params = {
#    'is_training': cfg.TRAIN.BN_TRAIN and is_training,
#    'decay': batch_norm_decay,
#    'epsilon': batch_norm_epsilon,
#    'scale': batch_norm_scale,
#    'trainable': cfg.TRAIN.BN_TRAIN,
#    'updates_collections': tf.GraphKeys.UPDATE_OPS
#  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=tf.nn.relu) as arg_sc:
 #     normalizer_fn=slim.batch_norm,
 #     normalizer_params=batch_norm_params):
 #   with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
 #     return arg_sc
    return arg_sc

class vgg16(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)
    self._arch = 'vgg16'

  def build_network(self, sess, is_training=True):
    with arg_scope(vgg_arg_scope(is_training=is_training)):
      with tf.variable_scope('vgg_16'):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
          initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
          initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
          initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
          initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        with tf.variable_scope('conv1'):
            net = slim.conv2d(self._image, self._filter_num[0], [3, 3],
                              trainable=False, scope='conv1_1')  # 64
            self._act.append(net)
            net = slim.conv2d(net, self._filter_num[1], [3, 3],
                              trainable=False, scope='conv1_2')  # 64
            self._act.append(net)
        net = slim.max_pool2d(net, [2,2], padding = 'SAME', scope='pool1')
        with tf.variable_scope('conv2'):
            net = slim.conv2d(net, self._filter_num[2], [3, 3],
                              trainable = False, scope='conv2_1')  # 128
            self._act.append(net)
            net = slim.conv2d(net, self._filter_num[3], [3, 3],
                              trainable = False, scope='conv2_2')  # 128
            self._act.append(net)
        net = slim.max_pool2d(net, [2,2], padding = 'SAME', scope = 'pool2')
        with tf.variable_scope('conv3'):
            net = slim.conv2d(net, self._filter_num[4], [3, 3],
                              trainable = is_training, scope='conv3_1')  # 256
            self._act.append(net)
            net = slim.conv2d(net, self._filter_num[5], [3, 3],
                              trainable = is_training, scope='conv3_2')  # 256
            self._act.append(net)
            net = slim.conv2d(net, self._filter_num[6], [3, 3],
                              trainable = is_training, scope='conv3_3')  # 256
            self._act.append(net)
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
        with tf.variable_scope('conv4'):
            net = slim.conv2d(net, self._filter_num[7], [3, 3],
                              trainable = is_training, scope='conv4_1')  # 512
            self._act.append(net)
            net = slim.conv2d(net, self._filter_num[8], [3, 3],
                              trainable = is_training, scope='conv4_2')  # 512
            self._act.append(net)
            net = slim.conv2d(net, self._filter_num[9], [3, 3],
                              trainable = is_training, scope='conv4_3')  # 512
            self._act.append(net)
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
        with tf.variable_scope('conv5'):
            net = slim.conv2d(net, self._filter_num[10], [3, 3],
                              trainable = is_training, scope='conv5_1')  # 512
            self._act.append(net)
            net = slim.conv2d(net, self._filter_num[11], [3, 3],
                              trainable = is_training, scope='conv5_2')  # 512
            self._act.append(net)
            net = slim.conv2d(net, self._filter_num[12], [3, 3],
                              trainable = is_training, scope='conv5_3')  # 512
            self._act.append(net)

        self._act_summaries.append(net)
        self._layers['head'] = net
        # build the anchors for the image
        self._anchor_component()

        # rpn
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training,
                      weights_initializer=initializer, scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1],
                                    trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None,
                                    scope='rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2,
                                                      'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape,
                                                      "rpn_cls_prob_reshape")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape,
                                          self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1],
                                    trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None,
                                    scope='rpn_bbox_pred')
        if is_training:
          rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
          rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
          # Try to have a determinestic order for the computing graph, for reproducibility
          with tf.control_dependencies([rpn_labels]):
            rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
          if cfg.TEST.MODE == 'nms':
            rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
          elif cfg.TEST.MODE == 'top':
            rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
          else:
            raise NotImplementedError

        # rcnn
        if cfg.POOLING_MODE == 'crop':
          pool5 = self._crop_pool_layer(net, rois, "pool5")
        else:
          raise NotImplementedError

        pool5_flat = slim.flatten(pool5, scope='flatten')
        fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
        if is_training:
          fc6 = slim.dropout(fc6, keep_prob=0.5,
                            is_training=True, scope='dropout6')
        fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
        if is_training:
          fc7 = slim.dropout(fc7, keep_prob=0.5,
                            is_training=True, scope='dropout7')
        cls_score = slim.fully_connected(fc7, self._num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None,
                                         scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
                                         weights_initializer=initializer_bbox,
                                         trainable=is_training,
                                         activation_fn=None,
                                         scope='bbox_pred')

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions["rois"] = rois

        self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred
