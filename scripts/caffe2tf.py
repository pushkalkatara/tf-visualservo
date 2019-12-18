import numpy as np
import sys, os
import argparse
import tempfile

# Edit the paths as needed:
caffe_root = "/home/pushkalkatara/rrc/FlowNet"
sys.path.insert(0, caffe_root + 'python')

import caffe

PARAMS = {
    'conv1': 'conv/conv1',
    'conv2': 'conv/conv2',
    'conv3': 'conv/conv3',
    'conv3_1': 'conv/conv3_1',
    'conv4': 'conv/conv4',
    'conv4_1': 'conv/conv4_1',
    'conv5': 'conv/conv5',
    'conv5_1': 'conv/conv5_1',
    'conv6': 'conv/conv6',
    'conv6_1': 'conv/conv6_1',
    'Convolution1': 'conv/pred6',
    'deconv5': 'deconv/deconv5',
    'upsample_flow6to5': 'deconv/deconvflow6',
    'Convolution2': 'deconv/predict5',
    'deconv4': 'deconv/deconv4',
    'upsample_flow5to4': 'deconv/deconvflow5',
    'Convolution3': 'deconv/predict4',
    'deconv3': 'deconv/deconv3',
    'upsample_flow4to3': 'deconv/deconvflow4',
    'Convolution4': 'deconv/predict3',
    'deconv2': 'deconv/deconv2',
    'upsample_flow3to2': 'deconv/deconvflow3',
    'Convolution5': 'deconv/predict2'
}

file = '/home/pushkalkatara/rrc/visual_servo/model/flownet_step_e-4ssize30k_iter_75000.caffemodel'
proto_path = '/home/pushkalkatara/rrc/visual_servo/model/deployfnet.prototxt'

net = caffe.Net(proto_path, file, caffe.TEST)

out = {}

tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
print(tmp.name)

for(caffe_param, tf_param) in PARAMS.items():
    out[tf_param + '/weights'] = net.params[caffe_param][0].data.transpose((2, 3, 1, 0))
    out[tf_param + '/biases'] = net.params[caffe_param][1].data

#tricky part: connecting the tensor after last pooling layer with the
#first fully-connected layer
# reshape to caffe cnn format
# change to tensorflow format
# again reshape to fc format
print(net.params['Convolution5'][0].data.shape)
print(net.params['fc7'][0].data.shape)
temp = net.params['fc7'][0].data.reshape((4096, 2, 96, 128))
temp = temp.transpose((2, 3, 1, 0))
print(temp.shape)
out['deconv/fc7/kernel'] = temp.reshape((24576, 4096))
out['deconv/fc7/bias'] = net.params['fc7'][1].data

out['deconv/fc_pose_xyz/kernel'] = net.params['fc_pose_xyz'][0].data.transpose((1,0))
out['deconv/fc_pose_xyz/bias'] = net.params['fc_pose_xyz'][1].data

out['deconv/fc_pose_wpqr/kernel'] = net.params['fc_pose_wpqr'][0].data.transpose((1,0))
out['deconv/fc_pose_wpqr/bias'] = net.params['fc_pose_wpqr'][1].data

np.save(tmp, out)
