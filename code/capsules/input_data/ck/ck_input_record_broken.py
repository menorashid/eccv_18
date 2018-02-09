# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input utility functions for mnist and mnist multi.

Handles reading from single digit, shifted single digit, and multi digit mnist
dataset.
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import os
import random
import tensorflow as tf
import scipy.misc
import numpy as np
import random
import input_data.ck.util as util
import itertools

# class ck():
#   def __init__(self,data_dir,prepend):
#     filenames = util.readLinesFromFile(data_dir)
#     random.shuffle(filenames)
#     generator = itertools.cycle(filenames)
#     self.generator = generator
#     self.prepend = prepend

def _read_and_decode(line, image_dim=28, distort=False,
                     split='train'):
  """Reads a single record and converts it to a tensor.

  Args:
    filename_queue: Tensor Queue, list of input files.
    image_dim: Scalar, the height (and width) of the image in pixels.
    distort: Boolean, whether to distort the input or not.
    split: String, the split of the data (test or train) to read from.

  Returns:
    Dictionary of the (Image, label) and the image height.

  """
  
  
  print line
  prepend = '..'
  # prepend
  im_path, label  = line.split(' ')
  im_path = os.path.join(prepend,im_path)
  im = scipy.misc.imresize(scipy.misc.imread(im_path),(image_dim,image_dim)).astype(np.float32)
  im = im * 1/255.
  im = im[:,:,np.newaxis]
  label = int(label)

  return im, label

def inputs(data_dir,
           batch_size,
           split,
           num_targets,
           height=28,
           distort=False,
           batch_capacity=5000,
           validate=False,
           ):
  """Reads input data.

  Args:
    data_dir: Directory of the data.
    batch_size: Number of examples per returned batch.
    split: train or test
    num_targets: 1 digit or 2 digit dataset.
    height: image height.
    distort: whether to distort the input image.
    batch_capacity: the number of elements to prefetch in a batch.
    validate: If set use training-validation for training and validation for
      test.

  Returns:
    Dictionary of Batched features and labels.

  """
  
  filenames = util.readLinesFromFile(data_dir)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [data_dir], shuffle=(split == 'train'))

    batched_features = cover_upper(filename_queue,data_dir, batch_size,height, distort=distort, split=split)
    
    return batched_features

def cover_upper(q_obj,
          filenames,
           batch_size,
           height,
           distort,
           split
           ):
    
    dummy = q_obj.dequeue()

    ims = []
    labels = []
    filenames = util.readLinesFromFile(filenames)

    for idx_line, line in enumerate(filenames):
    # itertools.filenames[:batch_size]:  
    # while True:
      # line = self.generator.next()
      
      random.shuffle(filenames)
#     
      im, label = _read_and_decode(
            line, image_dim=height, distort=distort, split=split)
      ims.append(im)
      labels.append(label)
      
      if len(ims)==batch_size:
        break
    
    
    images = tf.convert_to_tensor(np.concatenate(ims,2),dtype = tf.float32)
    labels = tf.convert_to_tensor(np.array(labels),dtype=tf.int64)
    label_hot = tf.one_hot(labels,8)
    
    batched_features = {}
    batched_features['images']=images
    batched_features['labels']=label_hot
    batched_features['recons_label']=labels
    batched_features['recons_image']=images
    batched_features['height'] = height
    batched_features['depth'] = 1
    batched_features['num_targets'] = 1
    # num_targets
    batched_features['num_classes'] = 8

    return batched_features
    
