import os
import re
import time
from datetime import datetime
import tensorflow as tf
from data.data_loader import inputs
from check_fun import showdepth, showImagefromArray,showImageLable,trans3DsToImg,showImageLableCom,showImageJoints,showImageJointsandResults
from tf_fun import regression_mse, correlation, make_dir, \
    fine_tune_prepare_layers, ft_optimizer_list
from pose_evaluation import getMeanError,getMeanError_np,getMean_np,getMeanError_train
import numpy as np
import cPickle
from checkpoint import  list_variables

def save_result_image(images_np,images_coms,images_Ms,labels_np,images_results,cube_22,name,line=True):
    val_im = images_np[0].reshape([128, 128])
    com = images_coms[0]
    M=images_Ms[0]
    jtorig=labels_np
    jcrop = trans3DsToImg(jtorig,com,M)
    re_jtorig=images_results
    re_jcrop = trans3DsToImg(re_jtorig, com, M)

    showImageJointsandResults(val_im,jcrop,re_jcrop,save=True,imagename=name,line=line,allJoints=True)

def train_model(config,seqconfig):
    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    val_data = os.path.join(config.tfrecord_dir, config.val_tfrecords)
    with tf.device('/cpu:0'):
        train_images,train_labels,com3Ds,Ms = inputs(tfrecord_file = train_data,
                                           num_epochs=config.epochs,
                                           image_target_size = config.image_target_size,
                                           label_shape=config.num_classes,
                                           batch_size =config.train_batch)
        val_images, val_labels, val_com3Ds, val_Ms = inputs(tfrecord_file=val_data,
                                                            num_epochs=config.epochs,
                                                            image_target_size=config.image_target_size,
                                                            label_shape=config.num_classes,
                                                            batch_size=config.val_batch)
    for d in ['/gpu:0','/gpu:1']:
        with tf.device(d):
            with tf.variable_scope("cnn") as scope:
                print("create training graph:")
                model=dense_model_struct()
                model.build(train_images,config.num_classes,train_mode=True)
                loss=tf.nn.l2_loss(model.output-train_labels)
                if config.wd_penalty is None:
                    train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)
                else:
                    wd_l = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'biases' not in v.name]
                    loss_wd=loss+(config.wd_penalty * tf.add_n([tf.nn.l2_loss(x) for x in wd_l]))
                    train_op = tf.train.AdamOptimizer(1e-5).minimize(loss_wd)

                train_labels_shaped=tf.reshape(train_labels,[config.train_batch,config.num_classes/3,3])*seqconfig['cube'][2] / 2.
                train_results_shaped=tf.reshape(model.output,[config.train_batch,config.num_classes/3,3])*seqconfig['cube'][2] / 2.
                train_error =getMeanError_train(train_labels_shaped,train_results_shaped)

                print("using validation")
                scope.reuse_variables()
                val_model=dense_model_struct()
                val_model.build(val_images,config.num_classes,train_mode=False)

                val_labels_shaped = tf.reshape(val_labels, [config.val_batch, config.num_classes / 3, 3])*seqconfig['cube'][2] / 2.
                val_results_shaped = tf.reshape(val_model.output, [config.val_batch, config.num_classes / 3, 3])*seqconfig['cube'][2] / 2.
                val_error = getMeanError_train(val_labels_shaped, val_results_shaped)

                tf.summary.scalar("loss", loss)
                if config.wd_penalty is not None:
                    tf.summary.scalar("loss_wd", loss_wd)
                tf.summary.scalar("train error", train_error)
                tf.summary.scalar("validation error", val_error)
                summary_op = tf.summary.merge_all()
            saver = tf.train.Saver(tf.global_variables())

        # Initialize the graph
        gpuconfig = tf.ConfigProto()
        gpuconfig.gpu_options.allow_growth = True
        gpuconfig.allow_soft_placement = True
        first_v=True
        with tf.Session(config=gpuconfig) as sess:
            summary_writer = tf.summary.FileWriter(config.train_summaries, sess.graph)
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            step =0
            coord=tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while not coord.should_stop():
                    _,image_np,image_label,image_coms,image_Ms,tr_error,tr_loss,tr_loss_wd = sess.run([train_op,train_images,train_labels,com3Ds,Ms,train_error,loss,loss_wd])
                    print("step={},loss={},losswd={},error={} mm".format(step,tr_loss,tr_loss_wd,tr_error))

                    if step % 200 ==0:
                        val_image_np, val_image_label, val_image_coms, val_image_Ms,v_error= sess.run(
                            [val_images, val_labels, val_com3Ds, val_Ms,val_error])
                        print("     val error={} mm".format(v_error))

                        summary_str=sess.run(summary_op)
                        summary_writer.add_summary(summary_str,step)
                        # save the model checkpoint if it's the best yet
                        if first_v is True:
                            val_min = v_error
                            first_v = False
                        else:
                            if v_error < val_min:
                                saver.save(sess, os.path.join(
                                    config.model_output,
                                    'cnn_model' + str(step) +'.ckpt'), global_step=step)
                                # store the new max validation accuracy
                                val_min = v_error
                    # if (step > 0) and (step <2):
                    #
                    #     for b in range(config.train_batch):
                    #         im = image_np[b]
                    #         print("im shape:{}".format(im.shape))
                    #         image_com = image_coms[b]
                    #         image_M = image_Ms[b]
                    #         #print("shape of im:{}".format(im.shape))
                    #         jts = image_label[b]
                    #         im = im.reshape([128,128])
                    #         check_image_label(im,jts,image_com,image_M,seqconfig['cube'][2] / 2.,allJoints=True,line=True)
                    #
                    #     val_im = val_image_np[0]
                    #     print("val_im shape:{}".format(val_im.shape))
                    #     val_image_com = val_image_coms[0]
                    #     val_image_M = val_image_Ms[0]
                    #     # print("shape of im:{}".format(im.shape))
                    #     val_jts = val_image_label[0]
                    #     val_im = val_im.reshape([128, 128])
                    #     check_image_label(val_im, val_jts, val_image_com, val_image_M, seqconfig['cube'][2] / 2.,allJoints=True,line=True)

                    step += 1

            except tf.errors.OutOfRangeError:
                print("Done. Epoch limit reached.")
            finally:
                coord.request_stop()
            coord.join(threads)

def test_model(config,seqconfig):
    test_data = os.path.join(config.tfrecord_dir, config.test_tfrecords)
    with tf.device('/cpu:0'):
        images, labels, com3Ds, Ms = inputs(tfrecord_file=test_data,
                                            num_epochs=1,
                                            image_target_size=config.image_target_size,
                                            label_shape=config.num_classes,
                                            batch_size=1)
    with tf.device('/gpu:1'):
        with tf.variable_scope("cnn") as scope:
            model=dense_model_struct()
            model.build(images, config.num_classes, train_mode=False)
            labels_shaped = tf.reshape(labels, [config.num_classes / 3, 3]) * \
                                seqconfig['cube'][2] / 2.
            results_shaped = tf.reshape(model.output, [config.num_classes / 3, 3]) * \
                                 seqconfig['cube'][2] / 2.
            error = getMeanError(labels_shaped, results_shaped)

            # Initialize the graph
        gpuconfig = tf.ConfigProto()
        gpuconfig.gpu_options.allow_growth = True
        gpuconfig.allow_soft_placement = True
        saver = tf.train.Saver()

        with tf.Session(config=gpuconfig) as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            step=0
            joint_labels = []
            joint_results = []
            coord = tf.train.Coordinator()
            threads=tf.train.start_queue_runners(coord=coord)
            try:
                while not coord.should_stop():
                    checkpoints=tf.train.latest_checkpoint(config.model_output)
                    saver.restore(sess,checkpoints)
                    images_np,labels_np,results_np,images_coms,images_Ms,joint_error,labels_sp,results_sp=sess.run([\
                        images,labels,model.output,com3Ds,Ms,error,labels_shaped,results_shaped])
                    joint_labels.append(labels_sp)
                    joint_results.append(results_sp)
                    print("step={}, test error={} mm".format(step,joint_error))

                    if step==0:
                        sum_error=joint_error
                    else:
                        sum_error=sum_error+joint_error
                    if step%100 ==0:
                        result_name="results_com/dense/results/image_{}.png".format(step)
                        save_result_image(images_np,images_coms,images_Ms,labels_sp,results_sp,seqconfig['cube'][2] / 2.,result_name)
                    if joint_error >40:
                        result_name = "results_com/dense/bad/image_{}.png".format(step)
                        save_result_image(images_np, images_coms, images_Ms, labels_sp, results_sp, seqconfig['cube'][2] / 2.,
                              result_name)
                    step+=1
            except tf.errors.OutOfRangeError:
                print("Done.Epoch limit reached.")
            finally:
                coord.request_stop()
            coord.join(threads)
            print("load model from {}".format(checkpoints))
            print ("testing mean error is {}mm".format(sum_error / step))

            pickleCache = 'results_com/dense/cnn_result_cache.pkl'
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((joint_labels, joint_results), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            np_labels = np.asarray(joint_labels)
            np_results = np.asarray(joint_results)
            np_mean = getMeanError_np(np_labels, np_results)
            print np_mean

class dense_model_struct:
    def __init__(self,trainable=True):
        self.trainable=trainable
        self.data_dict = None
        self.var_dict = {}

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(self,depth,output_shape,batch_norm=None,train_mode=None):
        print ("multi-scale dense network")
        input_image = tf.identity(depth, name="lr_input")
        self.conv0=self.conv_layer(input_image,int(input_image.get_shape()[-1]),12,"conv_0",filter_size=3)
        self.pool0=self.max_pool(self.conv0,'pool_0')

        # layer 1
        self.conv1_1=self.conv_layer(self.pool0,12,16,"conv_1_1",filter_size=3)
        self.conv1_2=self.conv_layer(self.conv1_1,16,24,"conv_1_2",filter_size=3,stride=[1,2,2,1])
        self.conv1_3=self.conv_layer(self.conv1_2,24,32,"conv_1_3",filter_size=3,stride=[1,2,2,1])

        # layer 2
        # # scale 1
        self.conv2_1=self.conv_layer(self.conv1_1,16,24,"conv_2_1",filter_size=3)
        # # scale 2
        self.conv2_2_1=self.conv_layer(self.conv1_1,16,24,"conv_2_2_1",filter_size=3,stride=[1,2,2,1])
        self.conv2_2_2=self.conv_layer(self.conv1_2,24,32,"conv_2_2_2",filter_size=3)
        self.conv2_2=tf.concat([self.conv2_2_1,self.conv2_2_2],axis=-1,name="conv_2_2")
        # # scale 3
        self.conv2_3_2=self.conv_layer(self.conv1_2,24,32,"conv_2_3_2",filter_size=3,stride=[1,2,2,1])
        self.conv2_3_3=self.conv_layer(self.conv1_3,32,48,"conv_2_3_3",filter_size=3)
        self.conv2_3=tf.concat([self.conv2_3_2,self.conv2_3_3],axis=-1,name="conv_2_3")

        # layer 3
        # # scale 1
        self.conv3_1_in=tf.concat([self.conv1_1,self.conv2_1],axis=-1,name="conv_3_1_input")
        self.conv3_1_1x1=self.conv_layer(self.conv3_1_in,int(self.conv3_1_in.get_shape()[-1]),24,name="conv_3_1_1x1",filter_size=1)
        self.conv3_1=self.conv_layer(self.conv3_1_1x1,24,32,name="conv_3_1",filter_size=3)

        # # scale 2
        self.conv3_2_1x1_1=self.conv_layer(self.conv3_1_in,int(self.conv3_1_in.get_shape()[-1]),32,name="conv_3_2_1x1_1",filter_size=1)
        self.conv3_2_1=self.conv_layer(self.conv3_2_1x1_1,32,48,name="conv_3_2_1",filter_size=3,stride=[1,2,2,1])

        self.conv3_2_in=tf.concat([self.conv1_2,self.conv2_2],axis=-1,name="conv_3_2_input")
        self.conv3_2_1x1_2=self.conv_layer(self.conv3_2_in,int(self.conv3_2_in.get_shape()[-1]),32,name="conv_3_2_1x1_2",filter_size=1)
        self.conv3_2_2 = self.conv_layer(self.conv3_2_1x1_2, 32, 48, name="conv_3_2_2", filter_size=3)

        self.conv3_2=tf.concat([self.conv3_2_1,self.conv3_2_2],axis=-1,name="conv_3_2")

        # # scale 3
        self.conv3_3_1x1_2=self.conv_layer(self.conv3_2_in,int(self.conv3_2_in.get_shape()[-1]),48,name="conv_3_3_1x1_2",filter_size=1)
        self.conv3_3_2=self.conv_layer(self.conv3_3_1x1_2,48,64,name="conv_3_3_2",filter_size=3,stride=[1,2,2,1])

        self.conv3_3_in=tf.concat([self.conv1_3,self.conv2_3],axis=-1,name="conv_3_3_input")
        self.conv3_3_1x1_3 = self.conv_layer(self.conv3_3_in, int(self.conv3_3_in.get_shape()[-1]), 48,
                                             name="conv_3_3_1x1_3", filter_size=1)
        self.conv3_3_3 = self.conv_layer(self.conv3_3_1x1_3, 48, 64, name="conv_3_3_3", filter_size=3)

        self.conv3_3 = tf.concat([self.conv3_3_2, self.conv3_3_3], axis=-1, name="conv_3_3")

        # layer 4
        # # scale 1
        self.conv4_1_in = tf.concat([self.conv1_1, self.conv2_1,self.conv3_1], axis=-1, name="conv_4_1_input")
        self.conv4_1_1x1 = self.conv_layer(self.conv4_1_in, int(self.conv4_1_in.get_shape()[-1]), 32,
                                           name="conv_4_1_1x1", filter_size=1)
        self.conv4_1 = self.conv_layer(self.conv4_1_1x1, 32, 48, name="conv_4_1", filter_size=3)

        # # scale 2
        self.conv4_2_1x1_1 = self.conv_layer(self.conv4_1_in, int(self.conv4_1_in.get_shape()[-1]), 48,
                                             name="conv_4_2_1x1_1", filter_size=1)
        self.conv4_2_1 = self.conv_layer(self.conv4_2_1x1_1, 48, 64, name="conv_4_2_1", filter_size=3,
                                         stride=[1, 2, 2, 1])

        self.conv4_2_in = tf.concat([self.conv1_2, self.conv2_2,self.conv3_2], axis=-1, name="conv_4_2_input")
        self.conv4_2_1x1_2 = self.conv_layer(self.conv4_2_in, int(self.conv4_2_in.get_shape()[-1]), 48,
                                             name="conv_4_2_1x1_2", filter_size=1)
        self.conv4_2_2 = self.conv_layer(self.conv4_2_1x1_2, 48, 64, name="conv_4_2_2", filter_size=3)

        self.conv4_2 = tf.concat([self.conv4_2_1, self.conv4_2_2], axis=-1, name="conv_4_2")

        # # scale 3
        self.conv4_3_1x1_2 = self.conv_layer(self.conv4_2_in, int(self.conv4_2_in.get_shape()[-1]), 64,
                                             name="conv_4_3_1x1_2", filter_size=1)
        self.conv4_3_2 = self.conv_layer(self.conv4_3_1x1_2, 64, 96, name="conv_4_3_2", filter_size=3,
                                         stride=[1, 2, 2, 1])

        self.conv4_3_in = tf.concat([self.conv1_3, self.conv2_3,self.conv3_3], axis=-1, name="conv_4_3_input")
        self.conv4_3_1x1_3 = self.conv_layer(self.conv4_3_in, int(self.conv4_3_in.get_shape()[-1]), 64,
                                             name="conv_4_3_1x1_3", filter_size=1)
        self.conv4_3_3 = self.conv_layer(self.conv4_3_1x1_3, 64, 96, name="conv_4_3_3", filter_size=3)

        self.conv4_3 = tf.concat([self.conv4_3_2, self.conv4_3_3], axis=-1, name="conv_4_3")

        # layer 5
        # # scale 1
        self.conv5_1_in = tf.concat([self.conv1_1, self.conv2_1, self.conv3_1,self.conv4_1], axis=-1, name="conv_5_1_input")
        self.conv5_1_1x1 = self.conv_layer(self.conv5_1_in, int(self.conv5_1_in.get_shape()[-1]), 48,
                                           name="conv_5_1_1x1", filter_size=1)
        self.conv5_1 = self.conv_layer(self.conv5_1_1x1, 48,64, name="conv_5_1", filter_size=3)

        # # scale 2
        self.conv5_2_1x1_1 = self.conv_layer(self.conv5_1_in, int(self.conv5_1_in.get_shape()[-1]), 64,
                                             name="conv_5_2_1x1_1", filter_size=1)
        self.conv5_2_1 = self.conv_layer(self.conv5_2_1x1_1, 64, 96,name="conv_5_2_1", filter_size=3,
                                         stride=[1, 2, 2, 1])

        self.conv5_2_in = tf.concat([self.conv1_2, self.conv2_2, self.conv3_2,self.conv4_2], axis=-1, name="conv_5_2_input")
        self.conv5_2_1x1_2 = self.conv_layer(self.conv5_2_in, int(self.conv5_2_in.get_shape()[-1]), 64,
                                             name="conv_5_2_1x1_2", filter_size=1)
        self.conv5_2_2 = self.conv_layer(self.conv5_2_1x1_2, 64, 96, name="conv_5_2_2", filter_size=3)

        self.conv5_2 = tf.concat([self.conv5_2_1, self.conv5_2_2], axis=-1, name="conv_5_2")

        # # scale 3
        self.conv5_3_1x1_2 = self.conv_layer(self.conv5_2_in, int(self.conv5_2_in.get_shape()[-1]), 96,
                                             name="conv_5_3_1x1_2", filter_size=1)
        self.conv5_3_2 = self.conv_layer(self.conv5_3_1x1_2, 96,128, name="conv_5_3_2", filter_size=3,
                                         stride=[1, 2, 2, 1])

        self.conv5_3_in = tf.concat([self.conv1_3, self.conv2_3, self.conv3_3,self.conv4_3], axis=-1, name="conv_5_3_input")
        self.conv5_3_1x1_3 = self.conv_layer(self.conv5_3_in, int(self.conv5_3_in.get_shape()[-1]), 96,
                                             name="conv_5_3_1x1_3", filter_size=1)
        self.conv5_3_3 = self.conv_layer(self.conv5_3_1x1_3, 96,128,name="conv_5_3_3", filter_size=3)

        self.conv5_3 = tf.concat([self.conv5_3_2, self.conv5_3_3], axis=-1, name="conv_5_3")

        # layer 6
        # # scale 1
        self.conv6_1_in = tf.concat([self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1,self.conv5_1], axis=-1,
                                    name="conv_6_1_input")
        self.conv6_1_1x1 = self.conv_layer(self.conv6_1_in, int(self.conv6_1_in.get_shape()[-1]), 64,
                                           name="conv_6_1_1x1", filter_size=1)
        self.conv6_1 = self.conv_layer(self.conv6_1_1x1, 64,96,name="conv_6_1", filter_size=3)

        # # scale 2
        self.conv6_2_1x1_1 = self.conv_layer(self.conv6_1_in, int(self.conv6_1_in.get_shape()[-1]), 96,
                                             name="conv_6_2_1x1_1", filter_size=1)
        self.conv6_2_1 = self.conv_layer(self.conv6_2_1x1_1, 96,128,name="conv_6_2_1", filter_size=3,
                                         stride=[1, 2, 2, 1])

        self.conv6_2_in = tf.concat([self.conv1_2, self.conv2_2, self.conv3_2, self.conv4_2,self.conv5_2], axis=-1,
                                    name="conv_6_2_input")
        self.conv6_2_1x1_2 = self.conv_layer(self.conv6_2_in, int(self.conv6_2_in.get_shape()[-1]), 96,
                                             name="conv_6_2_1x1_2", filter_size=1)
        self.conv6_2_2 = self.conv_layer(self.conv6_2_1x1_2, 96,128, name="conv_6_2_2", filter_size=3)

        self.conv6_2 = tf.concat([self.conv6_2_1, self.conv6_2_2], axis=-1, name="conv_6_2")

        # # scale 3
        self.conv6_3_1x1_2 = self.conv_layer(self.conv6_2_in, int(self.conv6_2_in.get_shape()[-1]), 128,
                                             name="conv_6_3_1x1_2", filter_size=1)
        self.conv6_3_2 = self.conv_layer(self.conv6_3_1x1_2, 128,144,name="conv_6_3_2", filter_size=3,
                                         stride=[1, 2, 2, 1])

        self.conv6_3_in = tf.concat([self.conv1_3, self.conv2_3, self.conv3_3, self.conv4_3,self.conv5_3], axis=-1,
                                    name="conv_6_3_input")
        self.conv6_3_1x1_3 = self.conv_layer(self.conv6_3_in, int(self.conv6_3_in.get_shape()[-1]), 128,
                                             name="conv_6_3_1x1_3", filter_size=1)
        self.conv6_3_3 = self.conv_layer(self.conv6_3_1x1_3, 128,144,name="conv_6_3_3", filter_size=3)

        self.conv6_3 = tf.concat([self.conv6_3_2, self.conv6_3_3], axis=-1, name="conv_6_3")

        # pooling
        self.pool1=self.avg_pool(self.conv6_1,"pool_1")
        self.pool2=self.avg_pool(self.conv6_2,"pool_2")
        self.pool3 = self.avg_pool(self.conv6_3, "pool_3")

        # fc
        self.fc1_1=self.fc_layer(self.pool1,np.prod([int(x) for x in self.pool1.get_shape()[1:]]),512,"fc_1_1")
        self.relu1_1=tf.nn.relu(self.fc1_1)
        if train_mode==True:
            self.relu1_1=tf.nn.dropout(self.relu1_1,0.7)

        self.fc1_2 = self.fc_layer(self.pool2, np.prod([int(x) for x in self.pool2.get_shape()[1:]]), 512, "fc_1_2")
        self.relu1_2 = tf.nn.relu(self.fc1_2)
        if train_mode == True:
            self.relu1_2 = tf.nn.dropout(self.relu1_2, 0.7)

        self.fc1_3 = self.fc_layer(self.pool3, np.prod([int(x) for x in self.pool3.get_shape()[1:]]), 512, "fc_1_3")
        self.relu1_3 = tf.nn.relu(self.fc1_3)
        if train_mode == True:
            self.relu1_3 = tf.nn.dropout(self.relu1_3, 0.7)

        self.concat=tf.concat([self.relu1_1,self.relu1_2,self.relu1_3],axis=-1)
        self.fc2=self.fc_layer(self.concat,512*3,1024,"fc_2")
        self.relu2=tf.nn.relu(self.fc2)
        if train_mode == True:
            self.relu2 = tf.nn.dropout(self.relu2, 0.5)

        self.fc3=self.fc_layer(self.relu2,1024,512,"fc_3")
        self.relu3=tf.nn.relu(self.fc3)
        if train_mode==True:
            self.relu3=tf.nn.dropout(self.relu3,0.3)

        self.fc4=self.fc_layer(self.relu3,512,output_shape,"fc_4")
        self.output=tf.identity(self.fc4)





    def batchnorm(self, layer):
        m, v = tf.nn.moments(layer, [0])
        return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool_4(self, bottom, name):
        return tf.nn.max_pool(
            bottom, ksize=[1, 4, 4, 1],
            strides=[1, 4, 4, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(
            self, bottom, in_channels,
            out_channels, name, filter_size=3, batchnorm=None, stride=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            if batchnorm is not None:
                if name in batchnorm:
                    relu = self.batchnorm(relu)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(
            self, filter_size, in_channels, out_channels,
            name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [filter_size, filter_size, in_channels, out_channels],
                tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [filter_size, filter_size, in_channels, out_channels],
                0.0, 0.001)
        bias_init = tf.truncated_normal([out_channels], .0, .001)
        filters = self.get_var(weight_init, name, 0, name + "_filters")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [in_size, out_size],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [in_size, out_size], 0.0, 0.001)
        bias_init = tf.truncated_normal([out_size], .0, .001)
        weights = self.get_var(weight_init, name, 0, name + "_weights")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return weights, biases

    def get_var(
            self, initial_value, name, idx,
            var_name, in_size=None, out_size=None):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolian to numpy
            if type(value) is list:
                var = tf.get_variable(
                    name=var_name, shape=value[0], initializer=value[1])
            else:
                var = tf.get_variable(name=var_name, initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        return var