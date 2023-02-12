import time
import os
import sys

import numpy as np
import tensorflow as tf
from spektral.layers import ops
from spektral.utils import sparse
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

from simple_data import make_simple_dataset
from etc.utils import logdir, to_numpy


def loss_fn(X, X_pred):
    """
        compute loss
        :param X: ground truth
        :param X_pred: model output
        :return: loss
        """
    loss = tf.keras.losses.mean_squared_error(X, X_pred)
    loss = tf.reduce_mean(loss)
    return loss


def loss_rec_bce(A_pred, A_label):
    a_logit = tf.sparse.to_dense(A_pred)
    a_logit = tf.reshape(a_logit, [-1])
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    rec_loss = bce(A_label.astype("f4"), a_logit)
    return rec_loss


def loss_rec_soft(A_pred, A_label):
    a_logit = tf.sparse.to_dense(A_pred)
    a_logit = tf.reshape(a_logit, [-1])
    rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=a_logit, labels=A_label.astype("f4"))
    rec_loss = tf.reduce_mean(rec_loss)
    return rec_loss


def loss_rec_weight(A_pred, A_label, pos_weight):
    a_logit = tf.sparse.to_dense(A_pred)
    a_logit = tf.reshape(a_logit, [-1])
    rec_loss = tf.nn.weighted_cross_entropy_with_logits(logits=a_logit, labels=A_label, pos_weight=pos_weight)
    rec_loss = tf.reduce_mean(rec_loss)
    return rec_loss


def loss_KL(X, model_z_mean, model_z_log_std):
    kl_loss=(0.5 / len(X)) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model_z_log_std - tf.square(model_z_mean) - tf.square(tf.exp(model_z_log_std)),1))
    return kl_loss