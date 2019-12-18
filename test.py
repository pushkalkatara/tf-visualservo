########################################
# configuration
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import os
import cv2
import numpy as np
import struct
import scipy.io as sio
import model

start_number = 0
test_pairs_number = 64
use_gpu_1 = True
batch_size = 32
width = 512
height = 384
max_to_keep = 0
path = '/home/pushkalkatara/rrc/scripts/model.ckpt-0'
def main():
	img1 = cv2.imread('/home/pushkalkatara/rrc/tf-visualservo/heads/seq-01/frame-000000.color.png').astype(np.float32)
	img1 = cv2.resize(img1, (width, height), interpolation = cv2.INTER_LINEAR)
	img2 = cv2.imread('/home/pushkalkatara/rrc/tf-visualservo/heads/seq-01/frame-000045.color.png').astype(np.float32)
	img2 = cv2.resize(img2, (width, height), interpolation = cv2.INTER_LINEAR)

	m = model.Model()

	with tf.Session() as sess:
		m.saver.restore(sess, path)
		feed_dict = dict()
		feed_dict[m.x1] = img1
		feed_dict[m.x2] = img2
		fc_pose_xyz, fc_pose_wpqr = sess.run([m.fc_pose_xyz, m.fc_pose_wpqr])
		print(fc_pose_xyz, fc_pose_wpqr)

main()
