import tensorflow as tf
import numpy as np
import os
import cv2
from utils import *

def deg2rad(deg):
    return deg * (np.pi/180.0)

def read_and_decode(filename_queue,target_size,label_shape,data_augment=False):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = {
        'label':tf.FixedLenFeature([],tf.string),
        'image':tf.FixedLenFeature([],tf.string),
    })
    print('in read and decode')
    #convert from a scalor string tensor
    label = tf.decode_raw(features['label'],tf.float32)
    image = tf.decode_raw(features['image'],tf.float32)

    #Need to reconstruct channels first then transpose channels
    image = tf.reshape(image,np.asarray(target_size))
    label.set_shape(label_shape)
    print label_shape
    return label, image

def inputs(tfrecord_file,num_epochs,image_target_size,label_shape,batch_size,data_augment=False):
    print('in input!')
    with tf.name_scope('input'):
        if os.path.exists(tfrecord_file) is False:
            print("{} not exists".format(tfrecord_file))
        # returns a queue. adds a queue runner for the queue to the current graph's QUEUE_RUNNER
        filename_queue = tf.train.string_input_producer([tfrecord_file],num_epochs=num_epochs)
        label, image = read_and_decode(filename_queue = filename_queue,target_size=image_target_size,label_shape = label_shape,data_augment = data_augment)
        # return a list or dictionary. adds 1) a shuffling queue into which tensors are enqueued; 2) a dequeue_many operation to create batches
        # from the queue 3) a queue runner to QUEUE_RUNNER collection , to enqueue the tensors.
        data,labels = tf.train.shuffle_batch([image,label],batch_size=batch_size,num_threads=2,capacity=100+3*batch_size,min_after_dequeue=1)

    return data,labels
    #return image, label,com3D,M