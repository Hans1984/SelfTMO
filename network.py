import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.stats as st
import os
import shutil
import time
import random
import tensorflow.contrib.slim as slim
from tensorlayer.layers import *

# VGG19 net
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = tf.AUTO_REUSE,
           fc_conv_padding='VALID'):
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      return net, end_points

#define the vgg_loss
def VGG19_slim(input, type, reuse, scope):
    # Define the feature to extract according to the type of perceptual
    if type == 'VGG54':
        target_layer =  'vgg_19/conv5/conv5_4'
    elif type == 'VGG22':
        target_layer =  'vgg_19/conv2/conv2_2'
    elif type == 'VGG34':
        target_layer = 'vgg_19/conv3/conv3_4'
    elif type == 'VGG11':
        target_layer = 'vgg_19/conv1/conv1_1'
    elif type == 'VGG21':
        target_layer = 'vgg_19/conv2/conv2_1'
    elif type == 'VGG31':
        target_layer = 'vgg_19/conv3/conv3_1'
    elif type == 'VGG41':
        target_layer = 'vgg_19/conv4/conv4_1'
    elif type == 'VGG51':
        target_layer = 'vgg_19/conv5/conv5_1'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, is_training=False, reuse=reuse)
    output = output[target_layer]
    return output

def encoder(input, name = 'encoder_mul_exp', is_training=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        out_1 = tf.layers.conv2d(input, 16, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        out_1_relu = tf.nn.relu(out_1, name = 'relu_1')

        out_2 = tf.layers.conv2d(out_1_relu, 32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        out_2_relu = tf.nn.relu(out_2, name = 'relu_2')

        out_3 = tf.layers.conv2d(out_2_relu, 64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        out_3_relu = tf.nn.relu(out_3, name = 'relu_3')

    return out_3_relu


def decoder(input, img_1, img_2, img_3, name = 'decoder_mul_exp', is_training=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        sb, sy, sx, sc  = input.get_shape().as_list()

        out_1 = tf.layers.conv2d(input, 32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        out_1_relu = tf.nn.relu(out_1)

        out_2 = tf.layers.conv2d(out_1_relu, 16, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        out_2_relu = tf.nn.relu(out_2)

        out_3 = tf.layers.conv2d(out_2_relu, 3, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        out = out_3 + img_1 + img_2 + img_3
        out = tf.nn.sigmoid(out)

        return out, out_3

def tmo_net(input_1, input_2, input_3, name = 'tmo_net', is_training=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
       #encoder
        output_1 = encoder(input_1, name = 'encoder', is_training=is_training)
        output_2 = encoder(input_2, name = 'encoder', is_training=is_training)
        output_3 = encoder(input_3, name = 'encoder', is_training=is_training)
        out_concat = tf.concat([output_1, output_2, output_3], axis = 3)
       
       #fusion
        _, sy, sx, sc = out_concat.get_shape().as_list()
        out_gated_1 = tf.layers.conv2d(out_concat, sc, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        out_gated_2 = tf.layers.conv2d(out_gated_1, sc, kernel_size=(1, 1), strides=(1, 1), padding='SAME')

       #decoder        
        out_img, out_res = decoder(out_gated_2, input_1, input_2, input_3, name = 'decoder', is_training = is_training)

    return out_img

def Gaussian_kernel(nsig= 2, filter_size = 13):
    filter_size = filter_size
    print('sigma:\n')
    print(nsig)
    print('filter_size:\n')
    print(filter_size)
    interval = (2*nsig+1.)/(filter_size)
    ll = np.linspace(-nsig-interval/2., nsig+interval/2., filter_size+1)
    kern1d = np.diff(st.norm.cdf(ll))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = kernel.astype(np.float32)

    weights_g = np.zeros((filter_size, filter_size, 1, 1))
    weights_g[:, :, 0, 0] = kernel
    weights_g = weights_g.astype(np.float32)
    return weights_g

def feature_filtered(input_feature, sigma=2, kernel_size_num = 13, kernel_size_m = 3):
    print('kernel_size_num:\n')
    print(kernel_size_num)
    print('sigma:\n')
    print(sigma)
    sb, sy, sx, sc = input_feature.get_shape().as_list()
    kernel_size_num = kernel_size_num
    weights_g = Gaussian_kernel(nsig = sigma, filter_size = kernel_size_num)
    weights_g = tf.tile(weights_g, [1,1, sc, 1])
    kernel_size_num = tf.cast(kernel_size_num, dtype = tf.int32)
    weights_m = tf.ones((kernel_size_num, kernel_size_num, 1, 1), dtype = tf.float32)
    weights_m = tf.tile(weights_m, [1,1, sc, 1])

    sum_k = kernel_size_num * kernel_size_num
    sum_k = tf.cast(sum_k, dtype = tf.float32)
    
    ##Gaussian
    out_gaussian = tf.nn.depthwise_conv2d(input_feature, weights_g, [1, 1, 1, 1], padding='SAME')
    ##Box
    out_box_feature_square = tf.nn.depthwise_conv2d(tf.pow(input_feature, 2.0), weights_m/sum_k, [1, 1, 1, 1], padding = 'SAME')
    out_box_feature_mean = tf.nn.depthwise_conv2d(input_feature, weights_m/sum_k, [1, 1, 1, 1], padding = 'SAME')
    out_box_feature_mean_square = tf.pow(out_box_feature_mean, 2.0)
    out_box_feature_mean_square = tf.where(tf.equal(out_box_feature_mean_square, 0.0), 10e-8*tf.ones_like(out_box_feature_mean_square), out_box_feature_mean_square)
    diff = out_box_feature_square - out_box_feature_mean_square
    loacl_std = tf.sqrt(tf.abs(diff) + 10e-8)

    return out_gaussian, loacl_std, out_box_feature_mean

def local_mean_std(input_feature, sigma = 2, kernel_size_num = 13, kernel_size_m = 3):

    mean_local, std_local, mean_local_box = feature_filtered(input_feature, sigma= sigma, kernel_size_num = kernel_size_num, kernel_size_m = kernel_size_m)

    return mean_local, std_local, mean_local_box

##paper used
def sign_num_den(input, gamma, beta, sigma = 2, kernel_size_num = 13, kernel_size_m = 3):
    local_mean, local_std, local_mean_box = local_mean_std(input, sigma = sigma, kernel_size_num = kernel_size_num)

    #num
    gaussian_norm = (input - local_mean) / (tf.abs(local_mean) + 10e-8)
    msk = tf.where(tf.greater(gaussian_norm, 0.0), tf.ones_like(gaussian_norm), (-1)*tf.zeros_like(gaussian_norm))
    gaussian_norm = tf.where(tf.equal(gaussian_norm, 0.0), 10e-8*tf.ones_like(gaussian_norm), gaussian_norm)
    gaussian_norm = tf.pow((tf.abs(gaussian_norm)), gamma)
    norm_num = msk*gaussian_norm
    #den
    local_norm = local_std/(tf.abs(local_mean_box) + 10e-8)
    norm_den = tf.pow(local_norm, beta)
    norm_den = 1.0 + norm_den
    return norm_num, norm_den

def feature_contrast_masking(input, gamma, beta, sigma_num = 2, kernel_size_num = 13, kernel_size_den = 13):
    norm_num, norm_den = sign_num_den(input, gamma = gamma, beta = beta, sigma = sigma_num, kernel_size_num = kernel_size_num, kernel_size_m = kernel_size_den)
    out = norm_num / norm_den
    return out

def masking_loss(input_1, input_2, gamma = 0.5, beta = 0.5, sigma_num = 2.0, kernel_size_num = 13, kernel_size_den = 13):
    auto_loss_output = feature_contrast_masking(input_1, 1.0, beta, sigma_num = sigma_num, kernel_size_num = kernel_size_num, kernel_size_den = kernel_size_den)
    auto_loss_gt = feature_contrast_masking(input_2, gamma, beta, sigma_num = sigma_num, kernel_size_num = kernel_size_num, kernel_size_den = kernel_size_den)
    diff = tf.subtract(auto_loss_output, auto_loss_gt)
    diff_abs = tf.abs(diff)
    cost = tf.reduce_mean(diff_abs)
    return cost

def FCM_loss(input_1, input_2, gamma =0.5, beta=0.5, sigma_num = 2, kernel_size_num = 13, kernel_size_den = 13):
    x_1 = VGG19_slim(input_1, 'VGG11', reuse = tf.AUTO_REUSE, scope='vgg19_1/')
    gt_1 = VGG19_slim(input_2, 'VGG11', reuse = tf.AUTO_REUSE, scope='vgg19_2/')
    
    x_2 = VGG19_slim(input_1, 'VGG21', reuse = tf.AUTO_REUSE, scope='vgg19_3/')
    gt_2 = VGG19_slim(input_2, 'VGG21', reuse = tf.AUTO_REUSE, scope='vgg19_4/')

    x_3 = VGG19_slim(input_1, 'VGG31', reuse = tf.AUTO_REUSE, scope='vgg19_5/')
    gt_3 = VGG19_slim(input_2, 'VGG31', reuse = tf.AUTO_REUSE, scope='vgg19_6/')


    cost_1 = masking_loss(x_1, gt_1, gamma = gamma, beta = beta, sigma_num = sigma_num, kernel_size_num = kernel_size_num,  kernel_size_den = kernel_size_den)
    cost_2 = masking_loss(x_2, gt_2, gamma = gamma, beta = beta, sigma_num = sigma_num, kernel_size_num = kernel_size_num,  kernel_size_den = kernel_size_den)
    cost_3 = masking_loss(x_3, gt_3, gamma = gamma, beta = beta, sigma_num = sigma_num, kernel_size_num = kernel_size_num,  kernel_size_den = kernel_size_den)
    
    cost_all = (cost_3 + cost_2 + cost_1)/3.
    return cost_all
