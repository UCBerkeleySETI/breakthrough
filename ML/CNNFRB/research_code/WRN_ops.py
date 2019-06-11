#import skimage.io  # bug. need to import this before tensorflow
#import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training o


#tf.app.flags.DEFINE_integer('input_size', 224, "input image size")

#leaky relu
def lrelu(x, alpha=0.2, name="lrelu"):
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    return f1 * x + f2 * tf.abs(x)

activation = tf.nn.relu

version = np.asarray(tf.__version__.split('.')).astype(np.int32)

# def _imagenet_preprocess(rgb):
#     """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
#     red, green, blue = tf.split(3, 3, rgb * 255.0)
#     bgr = tf.concat(3, [blue, green, red])
#     bgr -= IMAGENET_MEAN_BGR
#     return bgr


def engineer(x, scheme=(2,2)):
    #engineer 4 features of different ave pool and max pools
    e1 = tf.nn.avg_pool(x, 
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')
    e2 = tf.nn.avg_pool(x, 
            ksize=[1, 2, 1, 1],
            strides=[1, 2, 1, 1],
            padding='SAME')
    e2 = tf.nn.max_pool(e2, 
            ksize=[1, 1, 2, 1],
            strides=[1, 1, 2, 1],
            padding='SAME')
    e3 = tf.nn.avg_pool(x, 
            ksize=[1, 1, 2, 1],
            strides=[1, 1, 2, 1],
            padding='SAME')
    e3 = tf.nn.max_pool(e3, 
            ksize=[1, 2, 1, 1],
            strides=[1, 2, 1, 1],
            padding='SAME')
    e4 = tf.nn.max_pool(x, 
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')
    x = tf.concat([e1,e2,e3,e4], axis=-1)
    x = lrelu(x)
    return x


def stack(x, c, dim=2):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c, dim=dim)
    return x


def block(x, c, dim=2):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']
    if c['bottleneck']:  #original resnet with bottleneck
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c, dim=dim)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c, dim=dim)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c, dim=dim)
            x = bn(x, c)

        with tf.variable_scope('shortcut'):
            if filters_out != filters_in or c['block_stride'] != 1:
                c['ksize'] = 1
                c['stride'] = c['block_stride']
                c['conv_filters_out'] = filters_out
                #shortcut = bn(shortcut, c)  #try turning this off
                shortcut = conv(shortcut, c, dim=dim)
            
        return activation(x + shortcut)
            
    else:  #preactivation without bottleneck, plus dropout
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            
            x = bn(x, c)
            x = activation(x)
            x = conv(x, c, dim=dim)

        with tf.variable_scope('dropout'):
            x = tf.nn.dropout(x, c['keep_prob'])

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = bn(x, c)
            x = activation(x)
            x = conv(x, c, dim=dim)

        with tf.variable_scope('shortcut'):
            if filters_out != filters_in or c['block_stride'] != 1:
                c['ksize'] = 1
                c['stride'] = c['block_stride']
                c['conv_filters_out'] = filters_out
                #shortcut = bn(shortcut, c)  #try turning this off
                shortcut = conv(shortcut, c, dim=dim)
                

        return x + shortcut
            




def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c, name='fc'):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    #weights_initializer = tf.truncated_normal_initializer(
        #stddev=FC_WEIGHT_STDDEV)
    weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases, name=name)
    return x

def fc_tensor(x, c, name='fc_tensor'):
    inshape = list(x.get_shape()[1:]) 
    num_units_out = c['fc_units_out']
    inshape.append(tf.Dimension(num_units_out))
    #weights_initializer = tf.truncated_normal_initializer(
        #stddev=FC_WEIGHT_STDDEV)
    weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    weights = _get_variable('weights',
                            shape=inshape,
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)

    axes_a = list(np.arange(1, len(inshape)))
    axes_b = list(np.arange(0, len(inshape)-1))

    x = tf.add(tf.tensordot(x, weights, axes=[axes_a, axes_b]), biases, name=name)
    return x

def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c, dim=2):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    if dim == 2:
        shape = [ksize, ksize, filters_in, filters_out]
    elif dim ==3:
        shape = [ksize, ksize, ksize, filters_in, filters_out]
    else:
        raise Exception("conv shape is {}".format(dim))
    #initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    initializer = tf.contrib.layers.xavier_initializer(uniform=True)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    if dim == 2:
        return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    elif dim == 3:
        return tf.nn.conv3d(x, weights, [1, stride, stride, stride, 1], padding='SAME')


def max_pool(x, ksize=3, stride=2, dim=2):
    if dim == 2:
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')
    elif dim ==3:
        return tf.nn.max_pool3d(x,
                          ksize=[1, ksize, ksize, ksize, 1],
                          strides=[1, stride, stride, stride, 1],
                          padding='SAME')

