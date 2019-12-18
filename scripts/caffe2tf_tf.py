import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import sys, os

parameters = np.load('/tmp/tmp4g8Zjx', allow_pickle=True)
parameters = parameters[()]

for(name, param) in parameters.iteritems():
    tf.Variable(param, name=name)
    print("Saving variable `" + name + "` of shape ", param.shape)


global_step = slim.get_or_create_global_step()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save_path = saver.save(sess, 'model.ckpt', global_step=global_step)
    print("Model saved in file: %s" % save_path)
