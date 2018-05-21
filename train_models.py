import os
import re
import time
from datetime import datetime
import tensorflow as tf
from data_loader import inputs
from check_fun import showdepth, showImagefromArray,showImageLable,trans3DsToImg,showImageLableCom,showImageJoints,showImageJointsandResults, joints3DToImg
from pose_evaluation import getMeanError,getMeanError_np,getMean_np,getMeanError_train
import numpy as np
import cPickle


def check_image_label(im, jts, com, M,cube_22,allJoints=False,line=False):
    relen=len(jts)/3
    jt = jts.reshape((relen, 3))
    jtorig = jt * cube_22
    jcrop = trans3DsToImg(jtorig, com, M)
    showImageJoints(im,jcrop,allJoints=allJoints,line=line)

def show_image(im,jnts,allJoints=False,line=False):
    #import ipdb; ipdb.set_trace();
    jnts2d = joints3DToImg(jnts)
    showImageJoints(im.squeeze(),jnts2d,allJoints=allJoints,line=line)

def test_input_full(config,seqconfig):
    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    val_data = os.path.join(config.tfrecord_dir, config.test_tfrecords)
    with tf.device('/cpu:0'):
        train_images,train_labels = inputs(tfrecord_file = train_data,
                                           num_epochs=config.epochs,
                                           image_target_size = config.image_orig_size,
                                           label_shape=config.num_classes,
                                           batch_size =config.train_batch,
                                                     data_augment=False)
        val_images, val_labels = inputs(tfrecord_file=val_data,
                                                            num_epochs=config.epochs,
                                                            image_target_size=config.image_orig_size,
                                                            label_shape=config.num_classes,
                                                            batch_size=1)
        label_shaped = tf.reshape(train_labels,[config.train_batch,config.num_classes/3,3])
        error = getMeanError(label_shaped,label_shaped)
        val_label_shaped = tf.reshape(val_labels, [1, config.num_classes/3, 3])
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        step =0
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                image_np,image_label,train_error = sess.run([train_images,train_labels,error])
                print step
                val_image_np, val_image_label, val_image_reshaped_label = sess.run([val_images, val_labels, val_label_shaped])


                if (step > 0) :#and (step <2):

                    # for b in range(config.train_batch):
                    #     im = image_np[b]
                    #     image_com = image_coms[b]
                    #     image_M = image_Ms[b]
                    #
                    #     jts = image_label[b]
                    #     print("shape of jts:{}".format(jts.shape))
                    #     im = im.reshape([128,128])
                    #     check_image_label(im,jts,image_com,image_M,seqconfig['cube'][2] / 2.,allJoints=True,line=False)

                    val_im = val_image_np[0]
                    print("val_im shape:{}".format(val_im.shape))
                    val_jts = val_image_reshaped_label[0]
                    #val_im = val_im.reshape([128, 128])
                    #check_image_label(val_im, val_jts, val_image_com, val_image_M, seqconfig['cube'][2] / 2.,allJoints=True,line=False)
                    show_image(val_im,val_jts)
                step += 1

        except tf.errors.OutOfRangeError:
            print("Done. Epoch limit reached.")
        finally:
            coord.request_stop()
        coord.join(threads)