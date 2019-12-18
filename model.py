import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np

class Model(object):
    def __init__(self):
        # Input Scale and Mean Subtraction in Data Preprocessing scripts
        self.x1 = tf.placeholder(tf.float32, [None, 384, 512, 3], name='x1')  # image1
        self.x2 = tf.placeholder(tf.float32, [None, 384, 512, 3], name='x2')  # image2
        self.output = tf.placeholder(tf.float32, [], name='output')
        with tf.variable_scope('conv'):
            concat1 = tf.concat([self.x1, self.x2], 3)
            conv1 = slim.conv2d(concat1, 64, [7, 7], 2, scope='conv1')
            relu1 = tf.nn.relu(conv1)
            conv2 = slim.conv2d(relu1, 128, [5, 5], 2, scope='conv2')
            relu2 = tf.nn.relu(conv2)
            conv3 = slim.conv2d(relu2, 256, [5, 5], 2, scope='conv3')
            relu3 = tf.nn.relu(conv3)
            conv3_1 = slim.conv2d(relu3, 256, [3, 3], 1, scope='conv3_1')
            relu4 = tf.nn.relu(conv3_1)
            conv4 = slim.conv2d(relu4, 512, [3, 3], 2, scope='conv4')
            relu5 = tf.nn.relu(conv4)
            conv4_1 = slim.conv2d(relu5, 512, [3, 3], 1, scope='conv4_1')
            relu6 = tf.nn.relu(conv4_1)
            conv5 = slim.conv2d(relu6, 512, [3, 3], 2, scope='conv5')
            relu7 = tf.nn.relu(conv5)
            conv5_1 = slim.conv2d(relu7, 512, [3, 3], 1, scope='conv5_1')
            relu8 = tf.nn.relu(conv5_1)
            conv6 = slim.conv2d(relu8, 1024, [3, 3], 2, scope='conv6')
            relu9 = tf.nn.relu(conv6)
            conv6_1 = slim.conv2d(relu9, 1024, [3, 3], 1, scope='conv6_1')
            relu10 = tf.nn.relu(conv6_1)
            predict6 = slim.conv2d(conv6_1, 2, [3, 3], 1, activation_fn=None, scope='pred6')

        with tf.variable_scope('deconv'):
            deconv5 = slim.conv2d_transpose(conv6_1, 512, [4, 4], 2, scope='deconv5')
            relu11 = tf.nn.relu(deconv5)
            deconvflow6 = slim.conv2d_transpose(predict6, 2, [4, 4], 2, 'SAME', scope='deconvflow6')
            concat5 = tf.concat([conv5_1, relu11, deconvflow6], 3, name='concat5')
            predict5 = slim.conv2d(concat5, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict5')

            deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], 2, 'SAME', scope='deconv4')
            relu12 = tf.nn.relu(deconv4)
            deconvflow5 = slim.conv2d_transpose(predict5, 2, [4, 4], 2, 'SAME', scope='deconvflow5')
            concat4 = tf.concat([conv4_1, relu12, deconvflow5], 3, name='concat4')
            predict4 = slim.conv2d(concat4, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict4')

            deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], 2, 'SAME', scope='deconv3')
            relu13 = tf.nn.relu(deconv3)
            deconvflow4 = slim.conv2d_transpose(predict4, 2, [4, 4], 2, 'SAME', scope='deconvflow4')
            concat3 = tf.concat([conv3_1, relu13, deconvflow4], 3, name='concat3')
            predict3 = slim.conv2d(concat3, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict3')

            deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], 2, 'SAME', scope='deconv2')
            relu14 = tf.nn.relu(deconv2)
            deconvflow3 = slim.conv2d_transpose(predict3, 2, [4, 4], 2, 'SAME', scope='deconvflow3')
            concat2 = tf.concat([conv2, relu14, deconvflow3], 3, name='concat2')
            predict2 = slim.conv2d(concat2, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict2')
            relu15 = tf.nn.relu(predict2)
            fc7 = tf.layers.dense(relu15, 4096, activation=tf.nn.relu,
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_initializer=tf.initializers.glorot_uniform(),
                                  name='fc7')

            relu16 = tf.nn.relu(fc7)
            dropout1 = tf.nn.dropout(relu16, 0.5)

            fc8 = tf.layers.dense(dropout1, 3,
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_initializer=tf.initializers.glorot_uniform(),
                                  name='fc_pose_xyz')

            fc9 = tf.layers.dense(fc8, 4,
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_initializer=tf.initializers.glorot_uniform(),
                                  name='fc_pose_wpqr')


        optimizer = tf.train.AdamOptimizer(self.output)
        #self.train_op = slim.learning.create_train_op(self.loss, optimizer)
        self.tvars = tf.trainable_variables()
        self.variables_names = [v.name for v in self.tvars]
        self.init = tf.initialize_all_variables()
        print(self.variables_names)
        self.saver = tf.train.Saver(max_to_keep=0)
        #self.tvars = tf.trainable_variables()
        #self.variables_names = [v.name for v in self.tvars]

    def mean_loss(self, gt, predict):
        loss = tf.reduce_mean(tf.abs(gt-predict))
        return loss
