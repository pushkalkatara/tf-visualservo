import numpy as np
import sys, os
import argparse
import tempfile

# Edit the paths as needed:
caffe_root = "/home/pushkalkatara/rrc/FlowNet"
sys.path.insert(0, caffe_root + 'python')

import caffe

file = '/home/pushkalkatara/rrc/visual_servo/model/flownet_step_e-4ssize30k_iter_75000.caffemodel'
proto_path = '/home/pushkalkatara/rrc/visual_servo/model/deployfnet.prototxt'

net = caffe.Net(proto_path, file, caffe.TEST)

print(net.params.items())
