from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf


# from input_data.cifar10 import cifar10_input
from input_data.mnist import mnist_input_record
from input_data.ck import ck_input_record_new
# from input_data.norb import norb_input_record
# from models import capsule_model
# from models import conv_model



def main():
    data_dir = '../../data/ck_96/train_test_files_tfrecords/train_0.tfrecords'
    batch_size = 10
    split = 'train'
    validate = False
    num_targets = 1

    sess = tf.Session()
    # features = mnist_input_record.inputs(
    #                 data_dir=data_dir,
    #                 batch_size=batch_size,
    #                 split=split,
    #                 num_targets=num_targets,
    #                 validate=validate)
    
    features = ck_input_record_new.inputs(
                    data_dir=data_dir,
                    batch_size=batch_size,
                    split=split,
                    validate=validate)
 


    features['images'] = tf.Print(features['images'], [features['images'].get_shape(),tf.reduce_min(features['images']),tf.reduce_max(features['images'])],'images: ')
    features['labels'] = tf.Print(features['labels'], [features['labels'].get_shape(),tf.reduce_min(features['labels']),tf.reduce_max(features['labels'])],'labels: ')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    sess.run(tf.reduce_min(features['labels']))
    sess.run(features['images'])
    
    coord.request_stop()
    coord.join(threads)
    sess.close()



  

if __name__=='__main__':
    main()