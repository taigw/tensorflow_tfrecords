# Created on Wed Oct 11 2017
#
# @author: Guotai Wang

import os
import nibabel
import numpy as np
import tensorflow as tf
from parse_config import parse_config
from data_generator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator


def save_array_as_nifty_volume(data, filename):
    # numpy data shape [D, H, W]
    # nifty image shape [W, H, W]
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)

def read_tfrecords_test(config_file):
    config = parse_config(config_file)
    config_tfrecords  = config['tfrecords']
    batch_size = config_tfrecords['batch_size']


    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(config_tfrecords)

        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    num_epochs = 5
    train_batches_per_epoch = 2 #int(np.floor(tr_data.data_size/batch_size))

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        # Loop over number of epochs
        total_step = 0
        for epoch in range(num_epochs):
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            # Initialize iterator with the training dataset
            sess.run(training_init_op)
            for step in range(train_batches_per_epoch):

                # get next batch of data
                [img_batch, label_batch] = sess.run(next_batch)
                img_0 = img_batch[0,:,:,:, 0]
                lab_0 = label_batch[0,:,:,:,0]

                print(epoch, step, img_0.shape, lab_0.shape)
                # save the images
                save_array_as_nifty_volume(img_0, './temp/img{0:}.nii'.format(total_step))
                save_array_as_nifty_volume(lab_0, './temp/lab{0:}.nii'.format(total_step))
                total_step = total_step + 1

if __name__ == '__main__':
    config_file = 'config/read_tfrecords.txt'
    read_tfrecords_test(config_file)
