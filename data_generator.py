# Created on Wed Oct 11 2017
#
# @author: Guotai Wang
"""Containes a helper class for image input pipelines in tensorflow."""
import tensorflow as tf
import numpy as np
from random import shuffle
from tensorflow.contrib.data import TFRecordDataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import random

class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """
    def __init__(self, config):
        """ Create a new ImageDataGenerator.
        
        Receives a configure dictionary
        """
        self.config = config
        self.__check_image_patch_shape()
        data = TFRecordDataset(self.config['tfrecords_filename'],"ZLIB")
        batch_size = self.config['batch_size']
        data = data.map(self._parse_function, num_threads=5,
                        output_buffer_size=20*batch_size)
        data = data.batch(batch_size)
        self.data = data

    def __check_image_patch_shape(self):
        data_shape   = self.config['data_shape']
        weight_shape = self.config['weight_shape']
        label_shape  = self.config['label_shape']
        assert(len(data_shape) == 4 and len(weight_shape) == 4 and len(label_shape) == 4)
        label_margin = []
        for i in range(3):
            assert(data_shape[i] == weight_shape[i])
            assert(data_shape[i] >= label_shape[i])
            margin = (data_shape[i] - label_shape[i]) % 2
            assert( margin == 0)
            label_margin.append(margin)
        label_margin.append(0)
        self.label_margin = label_margin

    def _parse_function(self, example_proto):
        keys_to_features = {
            'image_raw':tf.FixedLenFeature((), tf.string),
            'weight_raw':tf.FixedLenFeature((), tf.string),
            'label_raw':tf.FixedLenFeature((), tf.string),
            'image_shape_raw':tf.FixedLenFeature((), tf.string),
            'weight_shape_raw':tf.FixedLenFeature((), tf.string),
            'label_shape_raw':tf.FixedLenFeature((), tf.string)}
        # parse the data
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        image_shape  = tf.decode_raw(parsed_features['image_shape_raw'],  tf.int32)
        weight_shape = tf.decode_raw(parsed_features['weight_shape_raw'], tf.int32)
        label_shape  = tf.decode_raw(parsed_features['label_shape_raw'],  tf.int32)
        
        image_shape  = tf.reshape(image_shape,  [4])
        weight_shape = tf.reshape(weight_shape, [4])
        label_shape  = tf.reshape(label_shape,  [4])
        
        image_raw   = tf.decode_raw(parsed_features['image_raw'],  tf.float32)
        weight_raw  = tf.decode_raw(parsed_features['weight_raw'], tf.float32)
        label_raw   = tf.decode_raw(parsed_features['label_raw'],  tf.int32)
        
        image_raw  = tf.reshape(image_raw, image_shape)
        weight_raw = tf.reshape(weight_raw, weight_shape)
        label_raw  = tf.reshape(label_raw, label_shape)
        
        # preprocess (slice to fixed size)
        [img_slice, weight_slice, label_slice] = self.__random_sample_patch(
                image_raw, weight_raw, label_raw)
        return img_slice, label_slice
            
    def __random_sample_patch(self, img, weight, label):
        """Sample a patch from the image with a random position.
            The output size of img_slice and label_slice may not be the same. 
            image, weight and label are sampled with the same central voxel.
        """
        data_shape_in   = tf.shape(img)
        weight_shape_in = tf.shape(weight)
        label_shape_in  = tf.shape(label)
        
        data_shape_out  = tf.constant(self.config['data_shape'])
        weight_shape_out= tf.constant(self.config['weight_shape'])
        label_shape_out = tf.constant(self.config['label_shape'])
        label_margin    = tf.constant(self.label_margin)
        
        data_shape_sub = tf.subtract(data_shape_in, data_shape_out)
        r = tf.random_uniform([], 0, 1.0)
        img_begin = tf.cast(tf.cast(data_shape_sub, tf.float32) * r, tf.int32)
        img_begin = tf.multiply(img_begin, tf.constant([1, 1, 1, 0]))
        
        lab_begin = img_begin + label_margin
        lab_begin = tf.multiply(lab_begin, tf.constant([1, 1, 1, 0]))
        
        img_slice    = tf.slice(img, img_begin, data_shape_out)
        weight_slice = tf.slice(weight, img_begin, weight_shape_out)
        label_slice  = tf.slice(label, lab_begin, label_shape_out)
        return [img_slice, weight_slice, label_slice]
