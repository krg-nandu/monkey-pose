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
        label_shaped = tf.reshape(train_labels, [config.train_batch, config.num_classes / 3, 3])
        split_lable = tf.split(label_shaped, 36, axis=1)
        P_label_shaped = tf.concat(
            [split_lable[0], split_lable[1], split_lable[2], split_lable[3], split_lable[4], split_lable[5],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        R_label_shaped = tf.concat(
            [split_lable[6], split_lable[7], split_lable[8], split_lable[9], split_lable[10], split_lable[11],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        M_label_shaped = tf.concat(
            [split_lable[12], split_lable[13], split_lable[14], split_lable[15], split_lable[16], split_lable[17],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        I_label_shaped = tf.concat(
            [split_lable[18], split_lable[19], split_lable[20], split_lable[21], split_lable[22], split_lable[23],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        T_label_shaped = tf.concat(
            [split_lable[24], split_lable[25], split_lable[26], split_lable[27], split_lable[28],
             split_lable[29], split_lable[30], split_lable[31], split_lable[32], split_lable[33], split_lable[34],
             split_lable[35]], axis=1)
        P_label=tf.reshape(P_label_shaped,[config.train_batch,P_label_shaped.get_shape().as_list()[1]*3])
        R_label = tf.reshape(R_label_shaped, [config.train_batch, R_label_shaped.get_shape().as_list()[1] * 3])
        M_label = tf.reshape(M_label_shaped, [config.train_batch, M_label_shaped.get_shape().as_list()[1] * 3])
        I_label = tf.reshape(I_label_shaped, [config.train_batch, I_label_shaped.get_shape().as_list()[1] * 3])
        T_label = tf.reshape(T_label_shaped, [config.train_batch, T_label_shaped.get_shape().as_list()[1] * 3])
        error = getMeanError(label_shaped, label_shaped)
        val_label_shaped = tf.reshape(val_labels, [config.val_batch, config.num_classes / 3, 3])
        val_split_lable = tf.split(val_label_shaped, 36, axis=1)
        val_P_label_shaped = tf.concat(
            [val_split_lable[0], val_split_lable[1], val_split_lable[2], val_split_lable[3], val_split_lable[4],
             val_split_lable[5],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_R_label_shaped = tf.concat(
            [val_split_lable[6], val_split_lable[7], val_split_lable[8], val_split_lable[9], val_split_lable[10],
             val_split_lable[11],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_M_label_shaped = tf.concat(
            [val_split_lable[12], val_split_lable[13], val_split_lable[14], val_split_lable[15], val_split_lable[16],
             val_split_lable[17],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_I_label_shaped = tf.concat(
            [val_split_lable[18], val_split_lable[19], val_split_lable[20], val_split_lable[21], val_split_lable[22],
             val_split_lable[23],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_T_label_shaped = tf.concat(
            [val_split_lable[24], val_split_lable[25], val_split_lable[26], val_split_lable[27], val_split_lable[28],
             val_split_lable[29], val_split_lable[30], val_split_lable[31], val_split_lable[32], val_split_lable[33],
             val_split_lable[34],
             val_split_lable[35]], axis=1)
        val_P_label = tf.reshape(val_P_label_shaped, [config.val_batch, val_P_label_shaped.get_shape().as_list()[1] * 3])
        val_R_label = tf.reshape(val_R_label_shaped, [config.val_batch, val_R_label_shaped.get_shape().as_list()[1] * 3])
        val_M_label = tf.reshape(val_M_label_shaped, [config.val_batch, val_M_label_shaped.get_shape().as_list()[1] * 3])
        val_I_label = tf.reshape(val_I_label_shaped, [config.val_batch, val_I_label_shaped.get_shape().as_list()[1] * 3])
        val_T_label = tf.reshape(val_T_label_shaped, [config.val_batch, val_T_label_shaped.get_shape().as_list()[1] * 3])
    with tf.device('/gpu:0'):
        with tf.variable_scope("cnn") as scope:
            print("create training graph:")
            model=dense_hier_model_struct()
            model.build(train_images,config.num_classes,P_label.get_shape().as_list()[1],R_label.get_shape().as_list()[1],M_label.get_shape().as_list()[1], \
                        I_label.get_shape().as_list()[1],T_label.get_shape().as_list()[1],train_mode=True)
            hand_loss=tf.nn.l2_loss(model.output-train_labels)
            p_loss=tf.nn.l2_loss(model.p_output-P_label)
            r_loss = tf.nn.l2_loss(model.r_output - R_label)
            m_loss = tf.nn.l2_loss(model.m_output - M_label)
            i_loss = tf.nn.l2_loss(model.i_output - I_label)
            t_loss = tf.nn.l2_loss(model.t_output - T_label)
            loss=hand_loss+p_loss+r_loss+m_loss+i_loss+t_loss
            if config.wd_penalty is None:
                train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)
            else:
                wd_l = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'biases' not in v.name]
                loss_wd=loss+(config.wd_penalty * tf.add_n([tf.nn.l2_loss(x) for x in wd_l]))
                train_op = tf.train.AdamOptimizer(1e-5).minimize(loss_wd)

            train_results_shaped=tf.reshape(model.output,[config.train_batch,config.num_classes/3,3])
            train_error =getMeanError_train(label_shaped,train_results_shaped)*seqconfig['cube'][2] / 2.
            train_P_results_shaped=tf.reshape(model.p_output,[config.train_batch,P_label.get_shape().as_list()[1]/3,3])
            train_P_error = getMeanError_train(P_label_shaped, train_P_results_shaped) * seqconfig['cube'][2] / 2.
            train_R_results_shaped=tf.reshape(model.r_output,[config.train_batch,R_label.get_shape().as_list()[1]/3,3])
            train_R_error = getMeanError_train(R_label_shaped, train_R_results_shaped) * seqconfig['cube'][2] / 2.
            train_M_results_shaped = tf.reshape(model.m_output, [config.train_batch,M_label.get_shape().as_list()[1] / 3, 3])
            train_M_error = getMeanError_train(M_label_shaped, train_M_results_shaped) * seqconfig['cube'][2] / 2.
            train_I_results_shaped = tf.reshape(model.i_output, [config.train_batch,I_label.get_shape().as_list()[1] / 3, 3])
            train_I_error = getMeanError_train(I_label_shaped, train_I_results_shaped) * seqconfig['cube'][2] / 2.
            train_T_results_shaped = tf.reshape(model.t_output, [config.train_batch,T_label.get_shape().as_list()[1] / 3, 3])
            train_T_error = getMeanError_train(T_label_shaped, train_T_results_shaped) * seqconfig['cube'][2] / 2.

            print("using validation")
            scope.reuse_variables()
            val_model=dense_hier_model_struct()
            val_model.build(val_images,config.num_classes,val_P_label.get_shape().as_list()[1],val_R_label.get_shape().as_list()[1],val_M_label.get_shape().as_list()[1], \
                        val_I_label.get_shape().as_list()[1],val_T_label.get_shape().as_list()[1],train_mode=False)

            val_results_shaped = tf.reshape(val_model.output, [config.val_batch, config.num_classes / 3, 3])
            val_error = getMeanError_train(val_label_shaped, val_results_shaped)*seqconfig['cube'][2] / 2.
            val_P_results_shaped = tf.reshape(val_model.p_output, [config.val_batch,val_P_label.get_shape().as_list()[1] / 3, 3])
            val_P_error = getMeanError_train(val_P_label_shaped, val_P_results_shaped) * seqconfig['cube'][2] / 2.
            val_R_results_shaped = tf.reshape(val_model.r_output, [config.val_batch,val_R_label.get_shape().as_list()[1] / 3, 3])
            val_R_error = getMeanError_train(val_R_label_shaped, val_R_results_shaped) * seqconfig['cube'][2] / 2.
            val_M_results_shaped = tf.reshape(val_model.m_output, [config.val_batch,val_M_label.get_shape().as_list()[1] / 3, 3])
            val_M_error = getMeanError_train(val_M_label_shaped, val_M_results_shaped) * seqconfig['cube'][2] / 2.
            val_I_results_shaped = tf.reshape(val_model.i_output, [config.val_batch,val_I_label.get_shape().as_list()[1] / 3, 3])
            val_I_error = getMeanError_train(val_I_label_shaped, val_I_results_shaped) * seqconfig['cube'][2] / 2.
            val_T_results_shaped = tf.reshape(val_model.t_output, [config.val_batch,val_T_label.get_shape().as_list()[1] / 3, 3])
            val_T_error = getMeanError_train(val_T_label_shaped, val_T_results_shaped) * seqconfig['cube'][2] / 2.

            tf.summary.scalar("loss", loss)
            tf.summary.scalar("p_loss", p_loss)
            tf.summary.scalar("r_loss", r_loss)
            tf.summary.scalar("m_loss", m_loss)
            tf.summary.scalar("i_loss", i_loss)
            tf.summary.scalar("t_loss", t_loss)

            if config.wd_penalty is not None:
                tf.summary.scalar("loss_wd", loss_wd)
            tf.summary.scalar("train error", train_error)
            tf.summary.scalar("train P error", train_P_error)
            tf.summary.scalar("train R error", train_R_error)
            tf.summary.scalar("train M error", train_M_error)
            tf.summary.scalar("train I error", train_I_error)
            tf.summary.scalar("train T error", train_T_error)

            tf.summary.scalar("validation error", val_error)
            tf.summary.scalar("validation P error", val_P_error)
            tf.summary.scalar("validation R error", val_R_error)
            tf.summary.scalar("validation M error", val_M_error)
            tf.summary.scalar("validation I error", val_I_error)
            tf.summary.scalar("validation T error", val_T_error)

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
                _,image_np,image_label,image_coms,image_Ms,tr_error,tr_loss,tr_loss_wd,tr_P_error,tr_R_error,tr_M_error,\
                    tr_I_error,tr_T_error,tr_P_loss,tr_R_loss,tr_M_loss,tr_I_loss,tr_T_loss = \
                    sess.run([train_op,train_images,train_labels,com3Ds,Ms,train_error,loss,loss_wd,train_P_error,\
                              train_R_error,train_M_error,train_I_error,train_T_error,p_loss,r_loss,m_loss,i_loss,t_loss])
                print("step={},loss={},losswd={},ploss={},rloss={},,mloss={},iloss={},tloss={},error={} mm,perror={} mm,rerror={} mm,merror={} mm,ierror={} mm,terror={} mm"\
                      .format(step,tr_loss,tr_loss_wd,tr_P_loss,tr_R_loss,tr_M_loss,tr_I_loss,tr_T_loss,tr_error,tr_P_error,tr_R_error,
                              tr_M_error,tr_I_error,tr_T_error))

                if step % 200 ==0:
                    val_image_np, val_image_label, val_image_coms, val_image_Ms,v_error,v_P_error,v_R_error,v_M_error,v_I_error,v_T_error= sess.run(
                        [val_images, val_labels, val_com3Ds, val_Ms,val_error,val_P_error,val_R_error,val_M_error,val_I_error,val_T_error])
                    print("     val_error={} mm, val_P_error={} mm, val_R_error={} mm, val_M_error={} mm, val_I_error={} mm, val_T_error={} mm"\
                          .format(v_error,v_P_error,v_R_error,v_M_error,v_I_error,v_T_error))

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
                                'hier_model' + str(step) +'.ckpt'), global_step=step)
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
            model=dense_hier_model_struct()
            model.build(images, config.num_classes, 13*3,13*3,13*3,13*3,12*3,train_mode=False)
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
                    # print(labels_sp)
                    # print(results_sp)

                    if step==0:
                        sum_error=joint_error
                    else:
                        sum_error=sum_error+joint_error
                    if step%100 ==0:
                        result_name="results_com/dense_hier/results/image_{}.png".format(step)
                        save_result_image(images_np,images_coms,images_Ms,labels_sp,results_sp,seqconfig['cube'][2] / 2.,result_name)
                    if joint_error >40:
                        result_name = "results_com/dense_hier/bad/image_{}.png".format(step)
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

            pickleCache = 'results_com/dense_hier/cnn_result_cache.pkl'
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((joint_labels, joint_results), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            np_labels = np.asarray(joint_labels)
            np_results = np.asarray(joint_results)
            np_mean = getMeanError_np(np_labels, np_results)
            print np_mean

class dense_hier_model_struct:
    def __init__(self,trainable=True):
        self.trainable = trainable
        self.data_dict = None
        self.var_dict = {}
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(self,depth,output_shape,P_shape,R_shape,M_shape,I_shape,T_shape,batch_norm=None,train_mode=None):
        print("dense hierarchical cnn network")
        input_image = tf.identity(depth, name="lr_input")
        self.conv1 = self.conv_layer(input_image, int(input_image.get_shape().as_list()[-1]), 12, "conv_1",
                                     filter_size=3, batchnorm=batch_norm)
        self.pool1 = self.max_pool(self.conv1, 'pool_1')

        # dense block 1
        ## layer 1
        ### scale 1
        self.dense1_conv1_scale1=self.conv_layer(self.pool1,12,16,"dense_1_conv_1_scale_1",filter_size=3)
        ### scale 2
        self.dense1_conv1_scale2 = self.conv_layer(self.dense1_conv1_scale1, 16, 24, "dense_1_conv_1_scale_2", filter_size=3,stride=[1,2,2,1])
        ### scale 3
        self.dense1_conv1_scale3 = self.conv_layer(self.dense1_conv1_scale2, 24, 32, "dense_1_conv_1_scale_3", filter_size=3,
                                                   stride=[1, 2, 2, 1])
        ## layer 2
        ### scale 1
        self.dense1_conv2_scale1=self.conv_layer(self.dense1_conv1_scale1,16,24,"dense_1_conv_2_scale_1",filter_size=3)
        ### scale 2
        self.dense1_conv2_scale2_1=self.conv_layer(self.dense1_conv1_scale1,16,24,"dense_1_conv_2_scale_2_1",filter_size=3,stride=[1,2,2,1])
        self.dense1_conv2_scale2_2=self.conv_layer(self.dense1_conv1_scale2,24,32,"dense_1_conv_2_scale_2_2",filter_size=3)
        self.dense1_conv2_scale2=tf.concat([self.dense1_conv2_scale2_1,self.dense1_conv2_scale2_2],axis=-1,name="dense_1_conv_2_scale_2")
        ### scale 3
        self.dense1_conv2_scale3_2=self.conv_layer(self.dense1_conv1_scale2,24,32,"dense_1_conv_2_scale_3_2",filter_size=3,stride=[1,2,2,1])
        self.dense1_conv2_scale3_3=self.conv_layer(self.dense1_conv1_scale3,32,48,"dense_1_conv_2_scale_3_3",filter_size=3)
        self.dense1_conv2_scale3=tf.concat([self.dense1_conv2_scale3_2,self.dense1_conv2_scale3_3],axis=-1,name="dense_1_conv_2_scale_3")

        ## layer 3
        ### scale 1
        self.dense1_conv3_scale1_input=tf.concat([self.dense1_conv1_scale1,self.dense1_conv2_scale1],axis=-1,name="dense_1_conv_3_scale_1_input")
        self.dense1_conv3_scale1_1x1=self.conv_layer(self.dense1_conv3_scale1_input,int(self.dense1_conv3_scale1_input.get_shape()[-1]),24,name="dense_1_conv_3_scale_1_1x1",filter_size=1)
        self.dense1_conv3_scale1=self.conv_layer(self.dense1_conv3_scale1_1x1,24,32,name="dense_1_conv_3_scale_1",filter_size=3)

        ### scale 2
        self.dense1_conv3_scale2_1x1_1=self.conv_layer(self.dense1_conv3_scale1_input,int(self.dense1_conv3_scale1_input.get_shape()[-1]),32,name="dense_1_conv_3_scale_2_1x1_1",filter_size=1)
        self.dense1_conv3_scale2_1=self.conv_layer(self.dense1_conv3_scale2_1x1_1,32,48,name="dense_1_conv_3_scale_2_1",filter_size=3,stride=[1,2,2,1])

        self.dense1_conv3_scale2_input=tf.concat([self.dense1_conv1_scale2,self.dense1_conv2_scale2],axis=-1,name="dense_1_conv_3_scale_2_input")
        self.dense1_conv3_scale2_1x1_2=self.conv_layer(self.dense1_conv3_scale2_input,int(self.dense1_conv3_scale2_input.get_shape()[-1]),32,name="dense_1_conv_3_scale_2_1x1_2",filter_size=1)
        self.dense1_conv3_scale2_2=self.conv_layer(self.dense1_conv3_scale2_1x1_2,32,48,name="dense_1_conv_3_scale_2_2",filter_size=3)

        self.dense1_conv3_scale2=tf.concat([self.dense1_conv3_scale2_1,self.dense1_conv3_scale2_2],axis=-1,name="dense_1_conv_3_scale_2")

        ### scale 3
        self.dense1_conv3_scale3_1x1_2=self.conv_layer(self.dense1_conv3_scale2_input,int(self.dense1_conv3_scale2_input.get_shape()[-1]),48,name="dense_1_conv_3_scale_3_1x1_2",filter_size=1)
        self.dense1_conv3_scale3_2=self.conv_layer(self.dense1_conv3_scale3_1x1_2,48,64,name="dense_1_conv_3_scale_3_2",filter_size=3,stride=[1,2,2,1])

        self.dense1_conv3_scale3_input=tf.concat([self.dense1_conv1_scale3,self.dense1_conv2_scale3],axis=-1,name="dense_1_conv_3_scale_3_input")
        self.dense1_conv3_scale3_1x1_3=self.conv_layer(self.dense1_conv3_scale3_input,int(self.dense1_conv3_scale3_input.get_shape()[-1]),48,name="dense_1_conv_3_scale_3_1x1_3",filter_size=1)
        self.dense1_conv3_scale3_3=self.conv_layer(self.dense1_conv3_scale3_1x1_3,48,64,name="dense_1_conv_3_scale_3_3",filter_size=3)

        self.dense1_conv3_scale3=tf.concat([self.dense1_conv3_scale3_2,self.dense1_conv3_scale3_3],axis=-1,name="dense_1_conv_3_scale_3")

        ## layer 4
        ### scale 1
        self.dense1_conv4_scale1_input = tf.concat([self.dense1_conv1_scale1, self.dense1_conv2_scale1,self.dense1_conv3_scale1], axis=-1,
                                                   name="dense_1_conv_4_scale_1_input")
        self.dense1_conv4_scale1_1x1 = self.conv_layer(self.dense1_conv4_scale1_input,
                                                       int(self.dense1_conv4_scale1_input.get_shape()[-1]), 32,
                                                       name="dense_1_conv_4_scale_1_1x1", filter_size=1)
        self.dense1_conv4_scale1 = self.conv_layer(self.dense1_conv4_scale1_1x1, 32, 48,
                                                   name="dense_1_conv_4_scale_1", filter_size=3)

        ### scale 2
        self.dense1_conv4_scale2_1x1_1 = self.conv_layer(self.dense1_conv4_scale1_input,
                                                         int(self.dense1_conv4_scale1_input.get_shape()[-1]), 48,
                                                         name="dense_1_conv_4_scale_2_1x1_1", filter_size=1)
        self.dense1_conv4_scale2_1 = self.conv_layer(self.dense1_conv4_scale2_1x1_1, 48, 64,
                                                     name="dense_1_conv_4_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense1_conv4_scale2_input = tf.concat([self.dense1_conv1_scale2, self.dense1_conv2_scale2,self.dense1_conv3_scale2], axis=-1,
                                                   name="dense_1_conv_4_scale_2_input")
        self.dense1_conv4_scale2_1x1_2 = self.conv_layer(self.dense1_conv4_scale2_input,
                                                         int(self.dense1_conv4_scale2_input.get_shape()[-1]), 48,
                                                         name="dense_1_conv_4_scale_2_1x1_2", filter_size=1)
        self.dense1_conv4_scale2_2 = self.conv_layer(self.dense1_conv4_scale2_1x1_2, 48, 64,
                                                     name="dense_1_conv_4_scale_2_2", filter_size=3)

        self.dense1_conv4_scale2 = tf.concat([self.dense1_conv4_scale2_1, self.dense1_conv4_scale2_2], axis=-1,
                                             name="dense_1_conv_4_scale_2")

        ### scale 3
        self.dense1_conv4_scale3_1x1_2 = self.conv_layer(self.dense1_conv4_scale2_input,
                                                         int(self.dense1_conv4_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_1_conv_4_scale_3_1x1_2", filter_size=1)
        self.dense1_conv4_scale3_2 = self.conv_layer(self.dense1_conv4_scale3_1x1_2, 64, 96,
                                                     name="dense_1_conv_4_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense1_conv4_scale3_input = tf.concat([self.dense1_conv1_scale3, self.dense1_conv2_scale3,self.dense1_conv3_scale3], axis=-1,
                                                   name="dense_1_conv_4_scale_3_input")
        self.dense1_conv4_scale3_1x1_3 = self.conv_layer(self.dense1_conv4_scale3_input,
                                                         int(self.dense1_conv4_scale3_input.get_shape()[-1]), 64,
                                                         name="dense_1_conv_4_scale_3_1x1_3", filter_size=1)
        self.dense1_conv4_scale3_3 = self.conv_layer(self.dense1_conv4_scale3_1x1_3, 64, 96,
                                                     name="dense_1_conv_4_scale_3_3", filter_size=3)

        self.dense1_conv4_scale3 = tf.concat([self.dense1_conv4_scale3_2, self.dense1_conv4_scale3_3], axis=-1,
                                             name="dense_1_conv_4_scale_3")

        # transition 1
        self.tran1_conv1=self.conv_layer(self.dense1_conv4_scale1,int(self.dense1_conv4_scale1.get_shape()[-1]),16,name="tran_1_conv_1",filter_size=1)
        self.tran1_pool1=self.max_pool(self.tran1_conv1,name="tran_1_pool_1")

        self.tran1_conv2=self.conv_layer(self.dense1_conv4_scale2,int(self.dense1_conv4_scale2.get_shape()[-1]),24,name="tran_1_conv_2",filter_size=1)
        self.tran1_pool2=self.max_pool(self.tran1_conv2,name="tran_1_pool_2")

        self.tran1_conv3 = self.conv_layer(self.dense1_conv4_scale3, int(self.dense1_conv4_scale3.get_shape()[-1]), 32,
                                           name="tran_1_conv_3", filter_size=1)
        self.tran1_pool3 = self.max_pool(self.tran1_conv3, name="tran_1_pool_3")

        # [finger P, R]
        # dense block 2
        ## layer 1
        ### scale 1
        self.dense2_conv1_scale1 = self.conv_layer(self.tran1_pool1, 16, 24, "dense_2_conv_1_scale_1", filter_size=3)
        ### scale 2
        self.dense2_conv1_scale2 = self.conv_layer(self.tran1_pool2, 24, 32, "dense_2_conv_1_scale_2",filter_size=3)
        ### scale 3
        self.dense2_conv1_scale3 = self.conv_layer(self.tran1_pool3, 32, 48, "dense_2_conv_1_scale_3",filter_size=3)

        ## layer 2
        ### scale 1
        self.dense2_conv2_scale1 = self.conv_layer(self.dense2_conv1_scale1, 24, 32, "dense_2_conv_2_scale_1",
                                                   filter_size=3)
        ### scale 2
        self.dense2_conv2_scale2_1 = self.conv_layer(self.dense2_conv1_scale1, 24, 32, "dense_2_conv_2_scale_2_1",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense2_conv2_scale2_2 = self.conv_layer(self.dense2_conv1_scale2, 32, 48, "dense_2_conv_2_scale_2_2",
                                                     filter_size=3)
        self.dense2_conv2_scale2 = tf.concat([self.dense2_conv2_scale2_1, self.dense2_conv2_scale2_2], axis=-1,
                                             name="dense_2_conv_2_scale_2")
        ### scale 3
        self.dense2_conv2_scale3_2 = self.conv_layer(self.dense2_conv1_scale2, 32, 48, "dense_2_conv_2_scale_3_2",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense2_conv2_scale3_3 = self.conv_layer(self.dense2_conv1_scale3, 48, 64, "dense_2_conv_2_scale_3_3",
                                                     filter_size=3)
        self.dense2_conv2_scale3 = tf.concat([self.dense2_conv2_scale3_2, self.dense2_conv2_scale3_3], axis=-1,
                                             name="dense_2_conv_2_scale_3")

        ## layer 3
        ### scale 1
        self.dense2_conv3_scale1_input = tf.concat([self.dense2_conv1_scale1, self.dense2_conv2_scale1], axis=-1,
                                                   name="dense_2_conv_3_scale_1_input")
        self.dense2_conv3_scale1_1x1 = self.conv_layer(self.dense2_conv3_scale1_input,
                                                       int(self.dense2_conv3_scale1_input.get_shape()[-1]), 32,
                                                       name="dense_2_conv_3_scale_1_1x1", filter_size=1)
        self.dense2_conv3_scale1 = self.conv_layer(self.dense2_conv3_scale1_1x1, 32, 48,
                                                   name="dense_2_conv_3_scale_1", filter_size=3)

        ### scale 2
        self.dense2_conv3_scale2_1x1_1 = self.conv_layer(self.dense2_conv3_scale1_input,
                                                         int(self.dense2_conv3_scale1_input.get_shape()[-1]), 48,
                                                         name="dense_2_conv_3_scale_2_1x1_1", filter_size=1)
        self.dense2_conv3_scale2_1 = self.conv_layer(self.dense2_conv3_scale2_1x1_1, 48, 64,
                                                     name="dense_2_conv_3_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense2_conv3_scale2_input = tf.concat([self.dense2_conv1_scale2, self.dense2_conv2_scale2], axis=-1,
                                                   name="dense_2_conv_3_scale_2_input")
        self.dense2_conv3_scale2_1x1_2 = self.conv_layer(self.dense2_conv3_scale2_input,
                                                         int(self.dense2_conv3_scale2_input.get_shape()[-1]), 48,
                                                         name="dense_2_conv_3_scale_2_1x1_2", filter_size=1)
        self.dense2_conv3_scale2_2 = self.conv_layer(self.dense2_conv3_scale2_1x1_2, 48, 64,
                                                     name="dense_2_conv_3_scale_2_2", filter_size=3)

        self.dense2_conv3_scale2 = tf.concat([self.dense2_conv3_scale2_1, self.dense2_conv3_scale2_2], axis=-1,
                                             name="dense_2_conv_3_scale_2")

        ### scale 3
        self.dense2_conv3_scale3_1x1_2 = self.conv_layer(self.dense2_conv3_scale2_input,
                                                         int(self.dense2_conv3_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_2_conv_3_scale_3_1x1_2", filter_size=1)
        self.dense2_conv3_scale3_2 = self.conv_layer(self.dense2_conv3_scale3_1x1_2, 64, 96,
                                                     name="dense_2_conv_3_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense2_conv3_scale3_input = tf.concat([self.dense2_conv1_scale3, self.dense2_conv2_scale3], axis=-1,
                                                   name="dense_2_conv_3_scale_3_input")
        self.dense2_conv3_scale3_1x1_3 = self.conv_layer(self.dense2_conv3_scale3_input,
                                                         int(self.dense2_conv3_scale3_input.get_shape()[-1]), 64,
                                                         name="dense_2_conv_3_scale_3_1x1_3", filter_size=1)
        self.dense2_conv3_scale3_3 = self.conv_layer(self.dense2_conv3_scale3_1x1_3, 64, 96,
                                                     name="dense_2_conv_3_scale_3_3", filter_size=3)

        self.dense2_conv3_scale3 = tf.concat([self.dense2_conv3_scale3_2, self.dense2_conv3_scale3_3], axis=-1,
                                             name="dense_2_conv_3_scale_3")

        ## layer 4
        ### scale 1
        self.dense2_conv4_scale1_input = tf.concat(
            [self.dense2_conv1_scale1, self.dense2_conv2_scale1, self.dense2_conv3_scale1], axis=-1,
            name="dense_2_conv_4_scale_1_input")
        self.dense2_conv4_scale1_1x1 = self.conv_layer(self.dense2_conv4_scale1_input,
                                                       int(self.dense2_conv4_scale1_input.get_shape()[-1]), 48,
                                                       name="dense_2_conv_4_scale_1_1x1", filter_size=1)
        self.dense2_conv4_scale1 = self.conv_layer(self.dense2_conv4_scale1_1x1, 48, 64,
                                                   name="dense_2_conv_4_scale_1", filter_size=3)

        ### scale 2
        self.dense2_conv4_scale2_1x1_1 = self.conv_layer(self.dense2_conv4_scale1_input,
                                                         int(self.dense2_conv4_scale1_input.get_shape()[-1]), 64,
                                                         name="dense_2_conv_4_scale_2_1x1_1", filter_size=1)
        self.dense2_conv4_scale2_1 = self.conv_layer(self.dense2_conv4_scale2_1x1_1, 64, 96,
                                                     name="dense_2_conv_4_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense2_conv4_scale2_input = tf.concat(
            [self.dense2_conv1_scale2, self.dense2_conv2_scale2, self.dense2_conv3_scale2], axis=-1,
            name="dense_2_conv_4_scale_2_input")
        self.dense2_conv4_scale2_1x1_2 = self.conv_layer(self.dense2_conv4_scale2_input,
                                                         int(self.dense2_conv4_scale2_input.get_shape()[-1]),64,
                                                         name="dense_2_conv_4_scale_2_1x1_2", filter_size=1)
        self.dense2_conv4_scale2_2 = self.conv_layer(self.dense2_conv4_scale2_1x1_2, 64, 96,
                                                     name="dense_2_conv_4_scale_2_2", filter_size=3)

        self.dense2_conv4_scale2 = tf.concat([self.dense2_conv4_scale2_1, self.dense2_conv4_scale2_2], axis=-1,
                                             name="dense_2_conv_4_scale_2")

        ### scale 3
        self.dense2_conv4_scale3_1x1_2 = self.conv_layer(self.dense2_conv4_scale2_input,
                                                         int(self.dense2_conv4_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_2_conv_4_scale_3_1x1_2", filter_size=1)
        self.dense2_conv4_scale3_2 = self.conv_layer(self.dense2_conv4_scale3_1x1_2, 96, 128,
                                                     name="dense_2_conv_4_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense2_conv4_scale3_input = tf.concat(
            [self.dense2_conv1_scale3, self.dense2_conv2_scale3, self.dense2_conv3_scale3], axis=-1,
            name="dense_2_conv_4_scale_3_input")
        self.dense2_conv4_scale3_1x1_3 = self.conv_layer(self.dense2_conv4_scale3_input,
                                                         int(self.dense2_conv4_scale3_input.get_shape()[-1]), 96,
                                                         name="dense_2_conv_4_scale_3_1x1_3", filter_size=1)
        self.dense2_conv4_scale3_3 = self.conv_layer(self.dense2_conv4_scale3_1x1_3, 96,128,
                                                     name="dense_2_conv_4_scale_3_3", filter_size=3)

        self.dense2_conv4_scale3 = tf.concat([self.dense2_conv4_scale3_2, self.dense2_conv4_scale3_3], axis=-1,
                                             name="dense_2_conv_4_scale_3")

        # transition 2
        self.tran2_conv1=self.conv_layer(self.dense2_conv4_scale1,int(self.dense2_conv4_scale1.get_shape()[-1]),24,name="tran_2_conv_1",filter_size=1)
        self.tran2_pool1=self.max_pool(self.tran2_conv1,name="tran_2_pool_1")

        self.tran2_conv2=self.conv_layer(self.dense2_conv4_scale2,int(self.dense2_conv4_scale2.get_shape()[-1]),32,name="tran_2_conv_2",filter_size=1)
        self.tran2_pool2=self.max_pool(self.tran2_conv2,name="tran_2_pool_2")

        self.tran2_conv3 = self.conv_layer(self.dense2_conv4_scale3, int(self.dense2_conv4_scale3.get_shape()[-1]), 48,
                                           name="tran_2_conv_3", filter_size=1)
        self.tran2_pool3 = self.max_pool(self.tran2_conv3, name="tran_2_pool_3")

        #[finger P]
        # dense block 3
        ## layer 1
        ### scale 1
        self.dense3_conv1_scale1 = self.conv_layer(self.tran2_pool1, 24, 32, "dense_3_conv_1_scale_1", filter_size=3)
        ### scale 2
        self.dense3_conv1_scale2 = self.conv_layer(self.tran2_pool2, 32, 48, "dense_3_conv_1_scale_2", filter_size=3)
        ### scale 3
        self.dense3_conv1_scale3 = self.conv_layer(self.tran2_pool3, 48, 64, "dense_3_conv_1_scale_3", filter_size=3)

        ## layer 2
        ### scale 1
        self.dense3_conv2_scale1 = self.conv_layer(self.dense3_conv1_scale1, 32, 48, "dense_3_conv_2_scale_1",
                                                   filter_size=3)
        ### scale 2
        self.dense3_conv2_scale2_1 = self.conv_layer(self.dense3_conv1_scale1, 32, 48, "dense_3_conv_2_scale_2_1",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense3_conv2_scale2_2 = self.conv_layer(self.dense3_conv1_scale2, 48, 64, "dense_3_conv_2_scale_2_2",
                                                     filter_size=3)
        self.dense3_conv2_scale2 = tf.concat([self.dense3_conv2_scale2_1, self.dense3_conv2_scale2_2], axis=-1,
                                             name="dense_3_conv_2_scale_2")
        ### scale 3
        self.dense3_conv2_scale3_2 = self.conv_layer(self.dense3_conv1_scale2, 48, 64, "dense_3_conv_2_scale_3_2",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense3_conv2_scale3_3 = self.conv_layer(self.dense3_conv1_scale3, 64, 96, "dense_3_conv_2_scale_3_3",
                                                     filter_size=3)
        self.dense3_conv2_scale3 = tf.concat([self.dense3_conv2_scale3_2, self.dense3_conv2_scale3_3], axis=-1,
                                             name="dense_3_conv_2_scale_3")

        ## layer 3
        ### scale 1
        self.dense3_conv3_scale1_input = tf.concat([self.dense3_conv1_scale1, self.dense3_conv2_scale1], axis=-1,
                                                   name="dense_3_conv_3_scale_1_input")
        self.dense3_conv3_scale1_1x1 = self.conv_layer(self.dense3_conv3_scale1_input,
                                                       int(self.dense3_conv3_scale1_input.get_shape()[-1]), 48,
                                                       name="dense_3_conv_3_scale_1_1x1", filter_size=1)
        self.dense3_conv3_scale1 = self.conv_layer(self.dense3_conv3_scale1_1x1, 48, 64,
                                                   name="dense_3_conv_3_scale_1", filter_size=3)

        ### scale 2
        self.dense3_conv3_scale2_1x1_1 = self.conv_layer(self.dense3_conv3_scale1_input,
                                                         int(self.dense3_conv3_scale1_input.get_shape()[-1]), 64,
                                                         name="dense_3_conv_3_scale_2_1x1_1", filter_size=1)
        self.dense3_conv3_scale2_1 = self.conv_layer(self.dense3_conv3_scale2_1x1_1, 64, 96,
                                                     name="dense_3_conv_3_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense3_conv3_scale2_input = tf.concat([self.dense3_conv1_scale2, self.dense3_conv2_scale2], axis=-1,
                                                   name="dense_3_conv_3_scale_2_input")
        self.dense3_conv3_scale2_1x1_2 = self.conv_layer(self.dense3_conv3_scale2_input,
                                                         int(self.dense3_conv3_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_3_conv_3_scale_2_1x1_2", filter_size=1)
        self.dense3_conv3_scale2_2 = self.conv_layer(self.dense3_conv3_scale2_1x1_2, 64, 96,
                                                     name="dense_3_conv_3_scale_2_2", filter_size=3)

        self.dense3_conv3_scale2 = tf.concat([self.dense3_conv3_scale2_1, self.dense3_conv3_scale2_2], axis=-1,
                                             name="dense_3_conv_3_scale_2")

        ### scale 3
        self.dense3_conv3_scale3_1x1_2 = self.conv_layer(self.dense3_conv3_scale2_input,
                                                         int(self.dense3_conv3_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_3_conv_3_scale_3_1x1_2", filter_size=1)
        self.dense3_conv3_scale3_2 = self.conv_layer(self.dense3_conv3_scale3_1x1_2, 96, 128,
                                                     name="dense_3_conv_3_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense3_conv3_scale3_input = tf.concat([self.dense3_conv1_scale3, self.dense3_conv2_scale3], axis=-1,
                                                   name="dense_3_conv_3_scale_3_input")
        self.dense3_conv3_scale3_1x1_3 = self.conv_layer(self.dense3_conv3_scale3_input,
                                                         int(self.dense3_conv3_scale3_input.get_shape()[-1]), 96,
                                                         name="dense_3_conv_3_scale_3_1x1_3", filter_size=1)
        self.dense3_conv3_scale3_3 = self.conv_layer(self.dense3_conv3_scale3_1x1_3, 96, 128,
                                                     name="dense_3_conv_3_scale_3_3", filter_size=3)

        self.dense3_conv3_scale3 = tf.concat([self.dense3_conv3_scale3_2, self.dense3_conv3_scale3_3], axis=-1,
                                             name="dense_3_conv_3_scale_3")

        ## layer 4
        ### scale 1
        self.dense3_conv4_scale1_input = tf.concat(
            [self.dense3_conv1_scale1, self.dense3_conv2_scale1, self.dense3_conv3_scale1], axis=-1,
            name="dense_3_conv_4_scale_1_input")
        self.dense3_conv4_scale1_1x1 = self.conv_layer(self.dense3_conv4_scale1_input,
                                                       int(self.dense3_conv4_scale1_input.get_shape()[-1]), 64,
                                                       name="dense_3_conv_4_scale_1_1x1", filter_size=1)
        self.dense3_conv4_scale1 = self.conv_layer(self.dense3_conv4_scale1_1x1, 64, 96,
                                                   name="dense_3_conv_4_scale_1", filter_size=3)

        ### scale 2
        self.dense3_conv4_scale2_1x1_1 = self.conv_layer(self.dense3_conv4_scale1_input,
                                                         int(self.dense3_conv4_scale1_input.get_shape()[-1]), 96,
                                                         name="dense_3_conv_4_scale_2_1x1_1", filter_size=1)
        self.dense3_conv4_scale2_1 = self.conv_layer(self.dense3_conv4_scale2_1x1_1, 96, 128,
                                                     name="dense_3_conv_4_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense3_conv4_scale2_input = tf.concat(
            [self.dense3_conv1_scale2, self.dense3_conv2_scale2, self.dense3_conv3_scale2], axis=-1,
            name="dense_3_conv_4_scale_2_input")
        self.dense3_conv4_scale2_1x1_2 = self.conv_layer(self.dense3_conv4_scale2_input,
                                                         int(self.dense3_conv4_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_3_conv_4_scale_2_1x1_2", filter_size=1)
        self.dense3_conv4_scale2_2 = self.conv_layer(self.dense3_conv4_scale2_1x1_2, 96, 128,
                                                     name="dense_3_conv_4_scale_2_2", filter_size=3)

        self.dense3_conv4_scale2 = tf.concat([self.dense3_conv4_scale2_1, self.dense3_conv4_scale2_2], axis=-1,
                                             name="dense_3_conv_4_scale_2")

        ### scale 3
        self.dense3_conv4_scale3_1x1_2 = self.conv_layer(self.dense3_conv4_scale2_input,
                                                         int(self.dense3_conv4_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_3_conv_4_scale_3_1x1_2", filter_size=1)
        self.dense3_conv4_scale3_2 = self.conv_layer(self.dense3_conv4_scale3_1x1_2, 128, 164,
                                                     name="dense_3_conv_4_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense3_conv4_scale3_input = tf.concat(
            [self.dense3_conv1_scale3, self.dense3_conv2_scale3, self.dense3_conv3_scale3], axis=-1,
            name="dense_3_conv_4_scale_3_input")
        self.dense3_conv4_scale3_1x1_3 = self.conv_layer(self.dense3_conv4_scale3_input,
                                                         int(self.dense3_conv4_scale3_input.get_shape()[-1]),128,
                                                         name="dense_3_conv_4_scale_3_1x1_3", filter_size=1)
        self.dense3_conv4_scale3_3 = self.conv_layer(self.dense3_conv4_scale3_1x1_3, 128, 164,
                                                     name="dense_3_conv_4_scale_3_3", filter_size=3)

        self.dense3_conv4_scale3 = tf.concat([self.dense3_conv4_scale3_2, self.dense3_conv4_scale3_3], axis=-1,
                                             name="dense_3_conv_4_scale_3")

        ## layer 5
        ### scale 1
        self.dense3_conv5_scale1_input = tf.concat(
            [self.dense3_conv1_scale1, self.dense3_conv2_scale1, self.dense3_conv3_scale1,self.dense3_conv4_scale1], axis=-1,
            name="dense_3_conv_5_scale_1_input")
        self.dense3_conv5_scale1_1x1 = self.conv_layer(self.dense3_conv5_scale1_input,
                                                       int(self.dense3_conv5_scale1_input.get_shape()[-1]), 96,
                                                       name="dense_3_conv_5_scale_1_1x1", filter_size=1)
        self.dense3_conv5_scale1 = self.conv_layer(self.dense3_conv5_scale1_1x1, 96, 128,
                                                   name="dense_3_conv_5_scale_1", filter_size=3)

        ### scale 2
        self.dense3_conv5_scale2_1x1_1 = self.conv_layer(self.dense3_conv5_scale1_input,
                                                         int(self.dense3_conv5_scale1_input.get_shape()[-1]), 128,
                                                         name="dense_3_conv_5_scale_2_1x1_1", filter_size=1)
        self.dense3_conv5_scale2_1 = self.conv_layer(self.dense3_conv5_scale2_1x1_1, 128, 164,
                                                     name="dense_3_conv_5_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense3_conv5_scale2_input = tf.concat(
            [self.dense3_conv1_scale2, self.dense3_conv2_scale2, self.dense3_conv3_scale2,self.dense3_conv4_scale2], axis=-1,
            name="dense_3_conv_5_scale_2_input")
        self.dense3_conv5_scale2_1x1_2 = self.conv_layer(self.dense3_conv5_scale2_input,
                                                         int(self.dense3_conv5_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_3_conv_5_scale_2_1x1_2", filter_size=1)
        self.dense3_conv5_scale2_2 = self.conv_layer(self.dense3_conv5_scale2_1x1_2, 128, 164,
                                                     name="dense_3_conv_5_scale_2_2", filter_size=3)

        self.dense3_conv5_scale2 = tf.concat([self.dense3_conv5_scale2_1, self.dense3_conv5_scale2_2], axis=-1,
                                             name="dense_3_conv_5_scale_2")

        ### scale 3
        self.dense3_conv5_scale3_1x1_2 = self.conv_layer(self.dense3_conv5_scale2_input,
                                                         int(self.dense3_conv5_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_3_conv_5_scale_3_1x1_2", filter_size=1)
        self.dense3_conv5_scale3_2 = self.conv_layer(self.dense3_conv5_scale3_1x1_2, 164, 198,
                                                     name="dense_3_conv_5_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense3_conv5_scale3_input = tf.concat(
            [self.dense3_conv1_scale3, self.dense3_conv2_scale3, self.dense3_conv3_scale3,self.dense3_conv4_scale3], axis=-1,
            name="dense_3_conv_5_scale_3_input")
        self.dense3_conv5_scale3_1x1_3 = self.conv_layer(self.dense3_conv5_scale3_input,
                                                         int(self.dense3_conv5_scale3_input.get_shape()[-1]), 164,
                                                         name="dense_3_conv_5_scale_3_1x1_3", filter_size=1)
        self.dense3_conv5_scale3_3 = self.conv_layer(self.dense3_conv5_scale3_1x1_3, 164, 198,
                                                     name="dense_3_conv_5_scale_3_3", filter_size=3)

        self.dense3_conv5_scale3 = tf.concat([self.dense3_conv5_scale3_2, self.dense3_conv5_scale3_3], axis=-1,
                                             name="dense_3_conv_5_scale_3")

        ## layer 6
        ### scale 1
        self.dense3_conv6_scale1_input = tf.concat(
            [self.dense3_conv1_scale1, self.dense3_conv2_scale1, self.dense3_conv3_scale1, self.dense3_conv4_scale1, self.dense3_conv5_scale1],
            axis=-1,
            name="dense_3_conv_6_scale_1_input")
        self.dense3_conv6_scale1_1x1 = self.conv_layer(self.dense3_conv6_scale1_input,
                                                       int(self.dense3_conv6_scale1_input.get_shape()[-1]), 128,
                                                       name="dense_3_conv_6_scale_1_1x1", filter_size=1)
        self.dense3_conv6_scale1 = self.conv_layer(self.dense3_conv6_scale1_1x1, 128, 164,
                                                   name="dense_3_conv_6_scale_1", filter_size=3)

        ### scale 2
        self.dense3_conv6_scale2_1x1_1 = self.conv_layer(self.dense3_conv6_scale1_input,
                                                         int(self.dense3_conv6_scale1_input.get_shape()[-1]), 164,
                                                         name="dense_3_conv_6_scale_2_1x1_1", filter_size=1)
        self.dense3_conv6_scale2_1 = self.conv_layer(self.dense3_conv6_scale2_1x1_1, 164, 198,
                                                     name="dense_3_conv_6_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense3_conv6_scale2_input = tf.concat(
            [self.dense3_conv1_scale2, self.dense3_conv2_scale2, self.dense3_conv3_scale2, self.dense3_conv4_scale2,self.dense3_conv5_scale2],
            axis=-1,
            name="dense_3_conv_6_scale_2_input")
        self.dense3_conv6_scale2_1x1_2 = self.conv_layer(self.dense3_conv6_scale2_input,
                                                         int(self.dense3_conv6_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_3_conv_6_scale_2_1x1_2", filter_size=1)
        self.dense3_conv6_scale2_2 = self.conv_layer(self.dense3_conv6_scale2_1x1_2, 164, 196,
                                                     name="dense_3_conv_6_scale_2_2", filter_size=3)

        self.dense3_conv6_scale2 = tf.concat([self.dense3_conv6_scale2_1, self.dense3_conv6_scale2_2], axis=-1,
                                             name="dense_3_conv_6_scale_2")

        ### scale 3
        self.dense3_conv6_scale3_1x1_2 = self.conv_layer(self.dense3_conv6_scale2_input,
                                                         int(self.dense3_conv6_scale2_input.get_shape()[-1]), 198,
                                                         name="dense_3_conv_6_scale_3_1x1_2", filter_size=1)
        self.dense3_conv6_scale3_2 = self.conv_layer(self.dense3_conv6_scale3_1x1_2, 198, 230,
                                                     name="dense_3_conv_6_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense3_conv6_scale3_input = tf.concat(
            [self.dense3_conv1_scale3, self.dense3_conv2_scale3, self.dense3_conv3_scale3, self.dense3_conv4_scale3,self.dense3_conv5_scale3],
            axis=-1,
            name="dense_3_conv_6_scale_3_input")
        self.dense3_conv6_scale3_1x1_3 = self.conv_layer(self.dense3_conv6_scale3_input,
                                                         int(self.dense3_conv6_scale3_input.get_shape()[-1]), 198,
                                                         name="dense_3_conv_6_scale_3_1x1_3", filter_size=1)
        self.dense3_conv6_scale3_3 = self.conv_layer(self.dense3_conv6_scale3_1x1_3, 198, 230,
                                                     name="dense_3_conv_6_scale_3_3", filter_size=3)

        self.dense3_conv6_scale3 = tf.concat([self.dense3_conv6_scale3_2, self.dense3_conv6_scale3_3], axis=-1,
                                             name="dense_3_conv_6_scale_3")

        # p output
        self.pool_p1=self.max_pool(self.dense3_conv6_scale1,"pool_p_1")
        self.pool_p2=self.max_pool(self.dense3_conv6_scale2,"pool_p_2")
        self.pool_p3=self.max_pool(self.dense3_conv6_scale3,"pool_p_3")

        self.fc1_p1=self.fc_layer(self.pool_p1,np.prod([int(x) for x in self.pool_p1.get_shape()[1:]]),512,"fc_1_p_1")
        self.relu1_p1=tf.nn.relu(self.fc1_p1)
        if train_mode==True:
            self.relu1_p1=tf.nn.dropout(self.relu1_p1,0.7)

        self.fc1_p2 = self.fc_layer(self.pool_p2, np.prod([int(x) for x in self.pool_p2.get_shape()[1:]]), 512,
                                    "fc_1_p_2")
        self.relu1_p2 = tf.nn.relu(self.fc1_p2)
        if train_mode == True:
            self.relu1_p2 = tf.nn.dropout(self.relu1_p2, 0.7)

        self.fc1_p3 = self.fc_layer(self.pool_p3, np.prod([int(x) for x in self.pool_p3.get_shape()[1:]]), 512,
                                    "fc_1_p_3")
        self.relu1_p3 = tf.nn.relu(self.fc1_p3)
        if train_mode == True:
            self.relu1_p3 = tf.nn.dropout(self.relu1_p3, 0.7)

        self.concat_p=tf.concat([self.relu1_p1,self.relu1_p2,self.relu1_p3],axis=-1)
        self.fc2_p=self.fc_layer(self.concat_p,512*3,1024,"fc_2_p")
        self.relu2_p=tf.nn.relu(self.fc2_p)
        if train_mode==True:
            self.relu2_p=tf.nn.dropout(self.relu2_p,0.5)

        self.fc3_p=self.fc_layer(self.relu2_p,1024,1024,"fc_3_p")
        self.relu3_p=tf.nn.relu(self.fc3_p)
        if train_mode==True:
            self.relu3_p=tf.nn.dropout(self.relu3_p,0.3)

        self.fc4_p = self.fc_layer(self.relu3_p, 1024, P_shape, "fc_4_p")
        self.p_output=tf.identity(self.fc4_p,name="p_out_put")

        # [finger R]
        # dense block 4
        ## layer 1
        ### scale 1
        self.dense4_conv1_scale1 = self.conv_layer(self.tran2_pool1, 24, 32, "dense_4_conv_1_scale_1", filter_size=3)
        ### scale 2
        self.dense4_conv1_scale2 = self.conv_layer(self.tran2_pool2, 32, 48, "dense_4_conv_1_scale_2", filter_size=3)
        ### scale 3
        self.dense4_conv1_scale3 = self.conv_layer(self.tran2_pool3, 48, 64, "dense_4_conv_1_scale_3", filter_size=3)

        ## layer 2
        ### scale 1
        self.dense4_conv2_scale1 = self.conv_layer(self.dense4_conv1_scale1, 32, 48, "dense_4_conv_2_scale_1",
                                                   filter_size=3)
        ### scale 2
        self.dense4_conv2_scale2_1 = self.conv_layer(self.dense4_conv1_scale1, 32, 48, "dense_4_conv_2_scale_2_1",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense4_conv2_scale2_2 = self.conv_layer(self.dense4_conv1_scale2, 48, 64, "dense_4_conv_2_scale_2_2",
                                                     filter_size=3)
        self.dense4_conv2_scale2 = tf.concat([self.dense4_conv2_scale2_1, self.dense4_conv2_scale2_2], axis=-1,
                                             name="dense_4_conv_2_scale_2")
        ### scale 3
        self.dense4_conv2_scale3_2 = self.conv_layer(self.dense4_conv1_scale2, 48, 64, "dense_4_conv_2_scale_3_2",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense4_conv2_scale3_3 = self.conv_layer(self.dense4_conv1_scale3, 64, 96, "dense_4_conv_2_scale_3_3",
                                                     filter_size=3)
        self.dense4_conv2_scale3 = tf.concat([self.dense4_conv2_scale3_2, self.dense4_conv2_scale3_3], axis=-1,
                                             name="dense_4_conv_2_scale_3")

        ## layer 3
        ### scale 1
        self.dense4_conv3_scale1_input = tf.concat([self.dense4_conv1_scale1, self.dense4_conv2_scale1], axis=-1,
                                                   name="dense_4_conv_3_scale_1_input")
        self.dense4_conv3_scale1_1x1 = self.conv_layer(self.dense4_conv3_scale1_input,
                                                       int(self.dense4_conv3_scale1_input.get_shape()[-1]), 48,
                                                       name="dense_4_conv_3_scale_1_1x1", filter_size=1)
        self.dense4_conv3_scale1 = self.conv_layer(self.dense4_conv3_scale1_1x1, 48, 64,
                                                   name="dense_4_conv_3_scale_1", filter_size=3)

        ### scale 2
        self.dense4_conv3_scale2_1x1_1 = self.conv_layer(self.dense4_conv3_scale1_input,
                                                         int(self.dense4_conv3_scale1_input.get_shape()[-1]), 64,
                                                         name="dense_4_conv_3_scale_2_1x1_1", filter_size=1)
        self.dense4_conv3_scale2_1 = self.conv_layer(self.dense4_conv3_scale2_1x1_1, 64, 96,
                                                     name="dense_4_conv_3_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense4_conv3_scale2_input = tf.concat([self.dense4_conv1_scale2, self.dense4_conv2_scale2], axis=-1,
                                                   name="dense_4_conv_3_scale_2_input")
        self.dense4_conv3_scale2_1x1_2 = self.conv_layer(self.dense4_conv3_scale2_input,
                                                         int(self.dense4_conv3_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_4_conv_3_scale_2_1x1_2", filter_size=1)
        self.dense4_conv3_scale2_2 = self.conv_layer(self.dense4_conv3_scale2_1x1_2, 64, 96,
                                                     name="dense_4_conv_3_scale_2_2", filter_size=3)

        self.dense4_conv3_scale2 = tf.concat([self.dense4_conv3_scale2_1, self.dense4_conv3_scale2_2], axis=-1,
                                             name="dense_4_conv_3_scale_2")

        ### scale 3
        self.dense4_conv3_scale3_1x1_2 = self.conv_layer(self.dense4_conv3_scale2_input,
                                                         int(self.dense4_conv3_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_4_conv_3_scale_3_1x1_2", filter_size=1)
        self.dense4_conv3_scale3_2 = self.conv_layer(self.dense4_conv3_scale3_1x1_2, 96, 128,
                                                     name="dense_4_conv_3_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense4_conv3_scale3_input = tf.concat([self.dense4_conv1_scale3, self.dense4_conv2_scale3], axis=-1,
                                                   name="dense_4_conv_3_scale_3_input")
        self.dense4_conv3_scale3_1x1_3 = self.conv_layer(self.dense4_conv3_scale3_input,
                                                         int(self.dense4_conv3_scale3_input.get_shape()[-1]), 96,
                                                         name="dense_4_conv_3_scale_3_1x1_3", filter_size=1)
        self.dense4_conv3_scale3_3 = self.conv_layer(self.dense4_conv3_scale3_1x1_3, 96, 128,
                                                     name="dense_4_conv_3_scale_3_3", filter_size=3)

        self.dense4_conv3_scale3 = tf.concat([self.dense4_conv3_scale3_2, self.dense4_conv3_scale3_3], axis=-1,
                                             name="dense_4_conv_3_scale_3")

        ## layer 4
        ### scale 1
        self.dense4_conv4_scale1_input = tf.concat(
            [self.dense4_conv1_scale1, self.dense4_conv2_scale1, self.dense4_conv3_scale1], axis=-1,
            name="dense_4_conv_4_scale_1_input")
        self.dense4_conv4_scale1_1x1 = self.conv_layer(self.dense4_conv4_scale1_input,
                                                       int(self.dense4_conv4_scale1_input.get_shape()[-1]), 64,
                                                       name="dense_4_conv_4_scale_1_1x1", filter_size=1)
        self.dense4_conv4_scale1 = self.conv_layer(self.dense4_conv4_scale1_1x1, 64, 96,
                                                   name="dense_4_conv_4_scale_1", filter_size=3)

        ### scale 2
        self.dense4_conv4_scale2_1x1_1 = self.conv_layer(self.dense4_conv4_scale1_input,
                                                         int(self.dense4_conv4_scale1_input.get_shape()[-1]), 96,
                                                         name="dense_4_conv_4_scale_2_1x1_1", filter_size=1)
        self.dense4_conv4_scale2_1 = self.conv_layer(self.dense4_conv4_scale2_1x1_1, 96, 128,
                                                     name="dense_4_conv_4_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense4_conv4_scale2_input = tf.concat(
            [self.dense4_conv1_scale2, self.dense4_conv2_scale2, self.dense4_conv3_scale2], axis=-1,
            name="dense_4_conv_4_scale_2_input")
        self.dense4_conv4_scale2_1x1_2 = self.conv_layer(self.dense4_conv4_scale2_input,
                                                         int(self.dense4_conv4_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_4_conv_4_scale_2_1x1_2", filter_size=1)
        self.dense4_conv4_scale2_2 = self.conv_layer(self.dense4_conv4_scale2_1x1_2, 96, 128,
                                                     name="dense_4_conv_4_scale_2_2", filter_size=3)

        self.dense4_conv4_scale2 = tf.concat([self.dense4_conv4_scale2_1, self.dense4_conv4_scale2_2], axis=-1,
                                             name="dense_4_conv_4_scale_2")

        ### scale 3
        self.dense4_conv4_scale3_1x1_2 = self.conv_layer(self.dense4_conv4_scale2_input,
                                                         int(self.dense4_conv4_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_4_conv_4_scale_3_1x1_2", filter_size=1)
        self.dense4_conv4_scale3_2 = self.conv_layer(self.dense4_conv4_scale3_1x1_2, 128, 164,
                                                     name="dense_4_conv_4_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense4_conv4_scale3_input = tf.concat(
            [self.dense4_conv1_scale3, self.dense4_conv2_scale3, self.dense4_conv3_scale3], axis=-1,
            name="dense_4_conv_4_scale_3_input")
        self.dense4_conv4_scale3_1x1_3 = self.conv_layer(self.dense4_conv4_scale3_input,
                                                         int(self.dense4_conv4_scale3_input.get_shape()[-1]), 128,
                                                         name="dense_4_conv_4_scale_3_1x1_3", filter_size=1)
        self.dense4_conv4_scale3_3 = self.conv_layer(self.dense4_conv4_scale3_1x1_3, 128, 164,
                                                     name="dense_4_conv_4_scale_3_3", filter_size=3)

        self.dense4_conv4_scale3 = tf.concat([self.dense4_conv4_scale3_2, self.dense4_conv4_scale3_3], axis=-1,
                                             name="dense_4_conv_4_scale_3")

        ## layer 5
        ### scale 1
        self.dense4_conv5_scale1_input = tf.concat(
            [self.dense4_conv1_scale1, self.dense4_conv2_scale1, self.dense4_conv3_scale1, self.dense4_conv4_scale1],
            axis=-1,
            name="dense_4_conv_5_scale_1_input")
        self.dense4_conv5_scale1_1x1 = self.conv_layer(self.dense4_conv5_scale1_input,
                                                       int(self.dense4_conv5_scale1_input.get_shape()[-1]), 96,
                                                       name="dense_4_conv_5_scale_1_1x1", filter_size=1)
        self.dense4_conv5_scale1 = self.conv_layer(self.dense4_conv5_scale1_1x1, 96, 128,
                                                   name="dense_4_conv_5_scale_1", filter_size=3)

        ### scale 2
        self.dense4_conv5_scale2_1x1_1 = self.conv_layer(self.dense4_conv5_scale1_input,
                                                         int(self.dense4_conv5_scale1_input.get_shape()[-1]), 128,
                                                         name="dense_4_conv_5_scale_2_1x1_1", filter_size=1)
        self.dense4_conv5_scale2_1 = self.conv_layer(self.dense4_conv5_scale2_1x1_1, 128, 164,
                                                     name="dense_4_conv_5_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense4_conv5_scale2_input = tf.concat(
            [self.dense4_conv1_scale2, self.dense4_conv2_scale2, self.dense4_conv3_scale2, self.dense4_conv4_scale2],
            axis=-1,
            name="dense_4_conv_5_scale_2_input")
        self.dense4_conv5_scale2_1x1_2 = self.conv_layer(self.dense4_conv5_scale2_input,
                                                         int(self.dense4_conv5_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_4_conv_5_scale_2_1x1_2", filter_size=1)
        self.dense4_conv5_scale2_2 = self.conv_layer(self.dense4_conv5_scale2_1x1_2, 128, 164,
                                                     name="dense_4_conv_5_scale_2_2", filter_size=3)

        self.dense4_conv5_scale2 = tf.concat([self.dense4_conv5_scale2_1, self.dense4_conv5_scale2_2], axis=-1,
                                             name="dense_4_conv_5_scale_2")

        ### scale 3
        self.dense4_conv5_scale3_1x1_2 = self.conv_layer(self.dense4_conv5_scale2_input,
                                                         int(self.dense4_conv5_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_4_conv_5_scale_3_1x1_2", filter_size=1)
        self.dense4_conv5_scale3_2 = self.conv_layer(self.dense4_conv5_scale3_1x1_2, 164, 198,
                                                     name="dense_4_conv_5_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense4_conv5_scale3_input = tf.concat(
            [self.dense4_conv1_scale3, self.dense4_conv2_scale3, self.dense4_conv3_scale3, self.dense4_conv4_scale3],
            axis=-1,
            name="dense_4_conv_5_scale_3_input")
        self.dense4_conv5_scale3_1x1_3 = self.conv_layer(self.dense4_conv5_scale3_input,
                                                         int(self.dense4_conv5_scale3_input.get_shape()[-1]), 164,
                                                         name="dense_4_conv_5_scale_3_1x1_3", filter_size=1)
        self.dense4_conv5_scale3_3 = self.conv_layer(self.dense4_conv5_scale3_1x1_3, 164, 198,
                                                     name="dense_4_conv_5_scale_3_3", filter_size=3)

        self.dense4_conv5_scale3 = tf.concat([self.dense4_conv5_scale3_2, self.dense4_conv5_scale3_3], axis=-1,
                                             name="dense_4_conv_5_scale_3")

        ## layer 6
        ### scale 1
        self.dense4_conv6_scale1_input = tf.concat(
            [self.dense4_conv1_scale1, self.dense4_conv2_scale1, self.dense4_conv3_scale1, self.dense4_conv4_scale1,
             self.dense4_conv5_scale1],
            axis=-1,
            name="dense_4_conv_6_scale_1_input")
        self.dense4_conv6_scale1_1x1 = self.conv_layer(self.dense4_conv6_scale1_input,
                                                       int(self.dense4_conv6_scale1_input.get_shape()[-1]), 128,
                                                       name="dense_4_conv_6_scale_1_1x1", filter_size=1)
        self.dense4_conv6_scale1 = self.conv_layer(self.dense4_conv6_scale1_1x1, 128, 164,
                                                   name="dense_4_conv_6_scale_1", filter_size=3)

        ### scale 2
        self.dense4_conv6_scale2_1x1_1 = self.conv_layer(self.dense4_conv6_scale1_input,
                                                         int(self.dense4_conv6_scale1_input.get_shape()[-1]), 164,
                                                         name="dense_4_conv_6_scale_2_1x1_1", filter_size=1)
        self.dense4_conv6_scale2_1 = self.conv_layer(self.dense4_conv6_scale2_1x1_1, 164, 198,
                                                     name="dense_4_conv_6_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense4_conv6_scale2_input = tf.concat(
            [self.dense4_conv1_scale2, self.dense4_conv2_scale2, self.dense4_conv3_scale2, self.dense4_conv4_scale2,
             self.dense4_conv5_scale2],
            axis=-1,
            name="dense_4_conv_6_scale_2_input")
        self.dense4_conv6_scale2_1x1_2 = self.conv_layer(self.dense4_conv6_scale2_input,
                                                         int(self.dense4_conv6_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_4_conv_6_scale_2_1x1_2", filter_size=1)
        self.dense4_conv6_scale2_2 = self.conv_layer(self.dense4_conv6_scale2_1x1_2, 164, 196,
                                                     name="dense_4_conv_6_scale_2_2", filter_size=3)

        self.dense4_conv6_scale2 = tf.concat([self.dense4_conv6_scale2_1, self.dense4_conv6_scale2_2], axis=-1,
                                             name="dense_4_conv_6_scale_2")

        ### scale 3
        self.dense4_conv6_scale3_1x1_2 = self.conv_layer(self.dense4_conv6_scale2_input,
                                                         int(self.dense4_conv6_scale2_input.get_shape()[-1]), 198,
                                                         name="dense_4_conv_6_scale_3_1x1_2", filter_size=1)
        self.dense4_conv6_scale3_2 = self.conv_layer(self.dense4_conv6_scale3_1x1_2, 198, 230,
                                                     name="dense_4_conv_6_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense4_conv6_scale3_input = tf.concat(
            [self.dense4_conv1_scale3, self.dense4_conv2_scale3, self.dense4_conv3_scale3, self.dense4_conv4_scale3,
             self.dense4_conv5_scale3],
            axis=-1,
            name="dense_4_conv_6_scale_3_input")
        self.dense4_conv6_scale3_1x1_3 = self.conv_layer(self.dense4_conv6_scale3_input,
                                                         int(self.dense4_conv6_scale3_input.get_shape()[-1]), 198,
                                                         name="dense_4_conv_6_scale_3_1x1_3", filter_size=1)
        self.dense4_conv6_scale3_3 = self.conv_layer(self.dense4_conv6_scale3_1x1_3, 198, 230,
                                                     name="dense_4_conv_6_scale_3_3", filter_size=3)

        self.dense4_conv6_scale3 = tf.concat([self.dense4_conv6_scale3_2, self.dense4_conv6_scale3_3], axis=-1,
                                             name="dense_4_conv_6_scale_3")

        # r output
        self.pool_r1 = self.max_pool(self.dense4_conv6_scale1, "pool_r_1")
        self.pool_r2 = self.max_pool(self.dense4_conv6_scale2, "pool_r_2")
        self.pool_r3 = self.max_pool(self.dense4_conv6_scale3, "pool_r_3")

        self.fc1_r1 = self.fc_layer(self.pool_r1, np.prod([int(x) for x in self.pool_r1.get_shape()[1:]]), 512,
                                    "fc_1_r_1")
        self.relu1_r1 = tf.nn.relu(self.fc1_r1)
        if train_mode == True:
            self.relu1_r1 = tf.nn.dropout(self.relu1_r1, 0.7)

        self.fc1_r2 = self.fc_layer(self.pool_r2, np.prod([int(x) for x in self.pool_r2.get_shape()[1:]]), 512,
                                    "fc_1_r_2")
        self.relu1_r2 = tf.nn.relu(self.fc1_r2)
        if train_mode == True:
            self.relu1_r2 = tf.nn.dropout(self.relu1_r2, 0.7)

        self.fc1_r3 = self.fc_layer(self.pool_r3, np.prod([int(x) for x in self.pool_r3.get_shape()[1:]]), 512,
                                    "fc_1_r_3")
        self.relu1_r3 = tf.nn.relu(self.fc1_r3)
        if train_mode == True:
            self.relu1_r3 = tf.nn.dropout(self.relu1_r3, 0.7)

        self.concat_r = tf.concat([self.relu1_r1, self.relu1_r2, self.relu1_r3], axis=-1)
        self.fc2_r = self.fc_layer(self.concat_r, 512 * 3, 1024, "fc_2_r")
        self.relu2_r = tf.nn.relu(self.fc2_r)
        if train_mode == True:
            self.relu2_r = tf.nn.dropout(self.relu2_r, 0.5)

        self.fc3_r = self.fc_layer(self.relu2_r, 1024, 1024, "fc_3_r")
        self.relu3_r = tf.nn.relu(self.fc3_r)
        if train_mode == True:
            self.relu3_r = tf.nn.dropout(self.relu3_r, 0.3)

        self.fc4_r = self.fc_layer(self.relu3_r, 1024, R_shape, "fc_4_r")
        self.r_output = tf.identity(self.fc4_r, name="r_out_put")

        # [finger M, I]
        # dense block 5
        ## layer 1
        ### scale 1
        self.dense5_conv1_scale1 = self.conv_layer(self.tran1_pool1, 16, 24, "dense_5_conv_1_scale_1", filter_size=3)
        ### scale 2
        self.dense5_conv1_scale2 = self.conv_layer(self.tran1_pool2, 24, 32, "dense_5_conv_1_scale_2", filter_size=3)
        ### scale 3
        self.dense5_conv1_scale3 = self.conv_layer(self.tran1_pool3, 32, 48, "dense_5_conv_1_scale_3", filter_size=3)

        ## layer 2
        ### scale 1
        self.dense5_conv2_scale1 = self.conv_layer(self.dense5_conv1_scale1, 24, 32, "dense_5_conv_2_scale_1",
                                                   filter_size=3)
        ### scale 2
        self.dense5_conv2_scale2_1 = self.conv_layer(self.dense5_conv1_scale1, 24, 32, "dense_5_conv_2_scale_2_1",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense5_conv2_scale2_2 = self.conv_layer(self.dense5_conv1_scale2, 32, 48, "dense_5_conv_2_scale_2_2",
                                                     filter_size=3)
        self.dense5_conv2_scale2 = tf.concat([self.dense5_conv2_scale2_1, self.dense5_conv2_scale2_2], axis=-1,
                                             name="dense_5_conv_2_scale_2")
        ### scale 3
        self.dense5_conv2_scale3_2 = self.conv_layer(self.dense5_conv1_scale2, 32, 48, "dense_5_conv_2_scale_3_2",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense5_conv2_scale3_3 = self.conv_layer(self.dense5_conv1_scale3, 48, 64, "dense_5_conv_2_scale_3_3",
                                                     filter_size=3)
        self.dense5_conv2_scale3 = tf.concat([self.dense5_conv2_scale3_2, self.dense5_conv2_scale3_3], axis=-1,
                                             name="dense_5_conv_2_scale_3")

        ## layer 3
        ### scale 1
        self.dense5_conv3_scale1_input = tf.concat([self.dense5_conv1_scale1, self.dense5_conv2_scale1], axis=-1,
                                                   name="dense_5_conv_3_scale_1_input")
        self.dense5_conv3_scale1_1x1 = self.conv_layer(self.dense5_conv3_scale1_input,
                                                       int(self.dense5_conv3_scale1_input.get_shape()[-1]), 32,
                                                       name="dense_5_conv_3_scale_1_1x1", filter_size=1)
        self.dense5_conv3_scale1 = self.conv_layer(self.dense5_conv3_scale1_1x1, 32, 48,
                                                   name="dense_5_conv_3_scale_1", filter_size=3)

        ### scale 2
        self.dense5_conv3_scale2_1x1_1 = self.conv_layer(self.dense5_conv3_scale1_input,
                                                         int(self.dense5_conv3_scale1_input.get_shape()[-1]), 48,
                                                         name="dense_5_conv_3_scale_2_1x1_1", filter_size=1)
        self.dense5_conv3_scale2_1 = self.conv_layer(self.dense5_conv3_scale2_1x1_1, 48, 64,
                                                     name="dense_5_conv_3_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense5_conv3_scale2_input = tf.concat([self.dense5_conv1_scale2, self.dense5_conv2_scale2], axis=-1,
                                                   name="dense_5_conv_3_scale_2_input")
        self.dense5_conv3_scale2_1x1_2 = self.conv_layer(self.dense5_conv3_scale2_input,
                                                         int(self.dense5_conv3_scale2_input.get_shape()[-1]), 48,
                                                         name="dense_5_conv_3_scale_2_1x1_2", filter_size=1)
        self.dense5_conv3_scale2_2 = self.conv_layer(self.dense5_conv3_scale2_1x1_2, 48, 64,
                                                     name="dense_5_conv_3_scale_2_2", filter_size=3)

        self.dense5_conv3_scale2 = tf.concat([self.dense5_conv3_scale2_1, self.dense5_conv3_scale2_2], axis=-1,
                                             name="dense_5_conv_3_scale_2")

        ### scale 3
        self.dense5_conv3_scale3_1x1_2 = self.conv_layer(self.dense5_conv3_scale2_input,
                                                         int(self.dense5_conv3_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_5_conv_3_scale_3_1x1_2", filter_size=1)
        self.dense5_conv3_scale3_2 = self.conv_layer(self.dense5_conv3_scale3_1x1_2, 64, 96,
                                                     name="dense_5_conv_3_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense5_conv3_scale3_input = tf.concat([self.dense5_conv1_scale3, self.dense5_conv2_scale3], axis=-1,
                                                   name="dense_5_conv_3_scale_3_input")
        self.dense5_conv3_scale3_1x1_3 = self.conv_layer(self.dense5_conv3_scale3_input,
                                                         int(self.dense5_conv3_scale3_input.get_shape()[-1]), 64,
                                                         name="dense_5_conv_3_scale_3_1x1_3", filter_size=1)
        self.dense5_conv3_scale3_3 = self.conv_layer(self.dense5_conv3_scale3_1x1_3, 64, 96,
                                                     name="dense_5_conv_3_scale_3_3", filter_size=3)

        self.dense5_conv3_scale3 = tf.concat([self.dense5_conv3_scale3_2, self.dense5_conv3_scale3_3], axis=-1,
                                             name="dense_5_conv_3_scale_3")

        ## layer 4
        ### scale 1
        self.dense5_conv4_scale1_input = tf.concat(
            [self.dense5_conv1_scale1, self.dense5_conv2_scale1, self.dense5_conv3_scale1], axis=-1,
            name="dense_5_conv_4_scale_1_input")
        self.dense5_conv4_scale1_1x1 = self.conv_layer(self.dense5_conv4_scale1_input,
                                                       int(self.dense5_conv4_scale1_input.get_shape()[-1]), 48,
                                                       name="dense_5_conv_4_scale_1_1x1", filter_size=1)
        self.dense5_conv4_scale1 = self.conv_layer(self.dense5_conv4_scale1_1x1, 48, 64,
                                                   name="dense_5_conv_4_scale_1", filter_size=3)

        ### scale 2
        self.dense5_conv4_scale2_1x1_1 = self.conv_layer(self.dense5_conv4_scale1_input,
                                                         int(self.dense5_conv4_scale1_input.get_shape()[-1]), 64,
                                                         name="dense_5_conv_4_scale_2_1x1_1", filter_size=1)
        self.dense5_conv4_scale2_1 = self.conv_layer(self.dense5_conv4_scale2_1x1_1, 64, 96,
                                                     name="dense_5_conv_4_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense5_conv4_scale2_input = tf.concat(
            [self.dense5_conv1_scale2, self.dense5_conv2_scale2, self.dense5_conv3_scale2], axis=-1,
            name="dense_5_conv_4_scale_2_input")
        self.dense5_conv4_scale2_1x1_2 = self.conv_layer(self.dense5_conv4_scale2_input,
                                                         int(self.dense5_conv4_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_5_conv_4_scale_2_1x1_2", filter_size=1)
        self.dense5_conv4_scale2_2 = self.conv_layer(self.dense5_conv4_scale2_1x1_2, 64, 96,
                                                     name="dense_5_conv_4_scale_2_2", filter_size=3)

        self.dense5_conv4_scale2 = tf.concat([self.dense5_conv4_scale2_1, self.dense5_conv4_scale2_2], axis=-1,
                                             name="dense_5_conv_4_scale_2")

        ### scale 3
        self.dense5_conv4_scale3_1x1_2 = self.conv_layer(self.dense5_conv4_scale2_input,
                                                         int(self.dense5_conv4_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_5_conv_4_scale_3_1x1_2", filter_size=1)
        self.dense5_conv4_scale3_2 = self.conv_layer(self.dense5_conv4_scale3_1x1_2, 96, 128,
                                                     name="dense_5_conv_4_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense5_conv4_scale3_input = tf.concat(
            [self.dense5_conv1_scale3, self.dense5_conv2_scale3, self.dense5_conv3_scale3], axis=-1,
            name="dense_5_conv_4_scale_3_input")
        self.dense5_conv4_scale3_1x1_3 = self.conv_layer(self.dense5_conv4_scale3_input,
                                                         int(self.dense5_conv4_scale3_input.get_shape()[-1]), 96,
                                                         name="dense_5_conv_4_scale_3_1x1_3", filter_size=1)
        self.dense5_conv4_scale3_3 = self.conv_layer(self.dense5_conv4_scale3_1x1_3, 96, 128,
                                                     name="dense_5_conv_4_scale_3_3", filter_size=3)

        self.dense5_conv4_scale3 = tf.concat([self.dense5_conv4_scale3_2, self.dense5_conv4_scale3_3], axis=-1,
                                             name="dense_5_conv_4_scale_3")

        # transition 3
        self.tran3_conv1 = self.conv_layer(self.dense5_conv4_scale1, int(self.dense5_conv4_scale1.get_shape()[-1]), 24,
                                           name="tran_3_conv_1", filter_size=1)
        self.tran3_pool1 = self.max_pool(self.tran3_conv1, name="tran_3_pool_1")

        self.tran3_conv2 = self.conv_layer(self.dense5_conv4_scale2, int(self.dense5_conv4_scale2.get_shape()[-1]), 32,
                                           name="tran_3_conv_2", filter_size=1)
        self.tran3_pool2 = self.max_pool(self.tran3_conv2, name="tran_3_pool_2")

        self.tran3_conv3 = self.conv_layer(self.dense5_conv4_scale3, int(self.dense5_conv4_scale3.get_shape()[-1]), 48,
                                           name="tran_3_conv_3", filter_size=1)
        self.tran3_pool3 = self.max_pool(self.tran3_conv3, name="tran_3_pool_3")

        # [finger M]
        # dense block 6
        ## layer 1
        ### scale 1
        self.dense6_conv1_scale1 = self.conv_layer(self.tran3_pool1, 24, 32, "dense_6_conv_1_scale_1", filter_size=3)
        ### scale 2
        self.dense6_conv1_scale2 = self.conv_layer(self.tran3_pool2, 32, 48, "dense_6_conv_1_scale_2", filter_size=3)
        ### scale 3
        self.dense6_conv1_scale3 = self.conv_layer(self.tran3_pool3, 48, 64, "dense_6_conv_1_scale_3", filter_size=3)

        ## layer 2
        ### scale 1
        self.dense6_conv2_scale1 = self.conv_layer(self.dense6_conv1_scale1, 32, 48, "dense_6_conv_2_scale_1",
                                                   filter_size=3)
        ### scale 2
        self.dense6_conv2_scale2_1 = self.conv_layer(self.dense6_conv1_scale1, 32, 48, "dense_6_conv_2_scale_2_1",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense6_conv2_scale2_2 = self.conv_layer(self.dense6_conv1_scale2, 48, 64, "dense_6_conv_2_scale_2_2",
                                                     filter_size=3)
        self.dense6_conv2_scale2 = tf.concat([self.dense6_conv2_scale2_1, self.dense6_conv2_scale2_2], axis=-1,
                                             name="dense_6_conv_2_scale_2")
        ### scale 3
        self.dense6_conv2_scale3_2 = self.conv_layer(self.dense6_conv1_scale2, 48, 64, "dense_6_conv_2_scale_3_2",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense6_conv2_scale3_3 = self.conv_layer(self.dense6_conv1_scale3, 64, 96, "dense_6_conv_2_scale_3_3",
                                                     filter_size=3)
        self.dense6_conv2_scale3 = tf.concat([self.dense6_conv2_scale3_2, self.dense6_conv2_scale3_3], axis=-1,
                                             name="dense_6_conv_2_scale_3")

        ## layer 3
        ### scale 1
        self.dense6_conv3_scale1_input = tf.concat([self.dense6_conv1_scale1, self.dense6_conv2_scale1], axis=-1,
                                                   name="dense_6_conv_3_scale_1_input")
        self.dense6_conv3_scale1_1x1 = self.conv_layer(self.dense6_conv3_scale1_input,
                                                       int(self.dense6_conv3_scale1_input.get_shape()[-1]), 48,
                                                       name="dense_6_conv_3_scale_1_1x1", filter_size=1)
        self.dense6_conv3_scale1 = self.conv_layer(self.dense6_conv3_scale1_1x1, 48, 64,
                                                   name="dense_6_conv_3_scale_1", filter_size=3)

        ### scale 2
        self.dense6_conv3_scale2_1x1_1 = self.conv_layer(self.dense6_conv3_scale1_input,
                                                         int(self.dense6_conv3_scale1_input.get_shape()[-1]), 64,
                                                         name="dense_6_conv_3_scale_2_1x1_1", filter_size=1)
        self.dense6_conv3_scale2_1 = self.conv_layer(self.dense6_conv3_scale2_1x1_1, 64, 96,
                                                     name="dense_6_conv_3_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense6_conv3_scale2_input = tf.concat([self.dense6_conv1_scale2, self.dense6_conv2_scale2], axis=-1,
                                                   name="dense_6_conv_3_scale_2_input")
        self.dense6_conv3_scale2_1x1_2 = self.conv_layer(self.dense6_conv3_scale2_input,
                                                         int(self.dense6_conv3_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_6_conv_3_scale_2_1x1_2", filter_size=1)
        self.dense6_conv3_scale2_2 = self.conv_layer(self.dense6_conv3_scale2_1x1_2, 64, 96,
                                                     name="dense_6_conv_3_scale_2_2", filter_size=3)

        self.dense6_conv3_scale2 = tf.concat([self.dense6_conv3_scale2_1, self.dense6_conv3_scale2_2], axis=-1,
                                             name="dense_6_conv_3_scale_2")

        ### scale 3
        self.dense6_conv3_scale3_1x1_2 = self.conv_layer(self.dense6_conv3_scale2_input,
                                                         int(self.dense6_conv3_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_6_conv_3_scale_3_1x1_2", filter_size=1)
        self.dense6_conv3_scale3_2 = self.conv_layer(self.dense6_conv3_scale3_1x1_2, 96, 128,
                                                     name="dense_6_conv_3_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense6_conv3_scale3_input = tf.concat([self.dense6_conv1_scale3, self.dense6_conv2_scale3], axis=-1,
                                                   name="dense_6_conv_3_scale_3_input")
        self.dense6_conv3_scale3_1x1_3 = self.conv_layer(self.dense6_conv3_scale3_input,
                                                         int(self.dense6_conv3_scale3_input.get_shape()[-1]), 96,
                                                         name="dense_6_conv_3_scale_3_1x1_3", filter_size=1)
        self.dense6_conv3_scale3_3 = self.conv_layer(self.dense6_conv3_scale3_1x1_3, 96, 128,
                                                     name="dense_6_conv_3_scale_3_3", filter_size=3)

        self.dense6_conv3_scale3 = tf.concat([self.dense6_conv3_scale3_2, self.dense6_conv3_scale3_3], axis=-1,
                                             name="dense_6_conv_3_scale_3")

        ## layer 4
        ### scale 1
        self.dense6_conv4_scale1_input = tf.concat(
            [self.dense6_conv1_scale1, self.dense6_conv2_scale1, self.dense6_conv3_scale1], axis=-1,
            name="dense_6_conv_4_scale_1_input")
        self.dense6_conv4_scale1_1x1 = self.conv_layer(self.dense6_conv4_scale1_input,
                                                       int(self.dense6_conv4_scale1_input.get_shape()[-1]), 64,
                                                       name="dense_6_conv_4_scale_1_1x1", filter_size=1)
        self.dense6_conv4_scale1 = self.conv_layer(self.dense6_conv4_scale1_1x1, 64, 96,
                                                   name="dense_6_conv_4_scale_1", filter_size=3)

        ### scale 2
        self.dense6_conv4_scale2_1x1_1 = self.conv_layer(self.dense6_conv4_scale1_input,
                                                         int(self.dense6_conv4_scale1_input.get_shape()[-1]), 96,
                                                         name="dense_6_conv_4_scale_2_1x1_1", filter_size=1)
        self.dense6_conv4_scale2_1 = self.conv_layer(self.dense6_conv4_scale2_1x1_1, 96, 128,
                                                     name="dense_6_conv_4_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense6_conv4_scale2_input = tf.concat(
            [self.dense6_conv1_scale2, self.dense6_conv2_scale2, self.dense6_conv3_scale2], axis=-1,
            name="dense_6_conv_4_scale_2_input")
        self.dense6_conv4_scale2_1x1_2 = self.conv_layer(self.dense6_conv4_scale2_input,
                                                         int(self.dense6_conv4_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_6_conv_4_scale_2_1x1_2", filter_size=1)
        self.dense6_conv4_scale2_2 = self.conv_layer(self.dense6_conv4_scale2_1x1_2, 96, 128,
                                                     name="dense_6_conv_4_scale_2_2", filter_size=3)

        self.dense6_conv4_scale2 = tf.concat([self.dense6_conv4_scale2_1, self.dense6_conv4_scale2_2], axis=-1,
                                             name="dense_6_conv_4_scale_2")

        ### scale 3
        self.dense6_conv4_scale3_1x1_2 = self.conv_layer(self.dense6_conv4_scale2_input,
                                                         int(self.dense6_conv4_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_6_conv_4_scale_3_1x1_2", filter_size=1)
        self.dense6_conv4_scale3_2 = self.conv_layer(self.dense6_conv4_scale3_1x1_2, 128, 164,
                                                     name="dense_6_conv_4_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense6_conv4_scale3_input = tf.concat(
            [self.dense6_conv1_scale3, self.dense6_conv2_scale3, self.dense6_conv3_scale3], axis=-1,
            name="dense_6_conv_4_scale_3_input")
        self.dense6_conv4_scale3_1x1_3 = self.conv_layer(self.dense6_conv4_scale3_input,
                                                         int(self.dense6_conv4_scale3_input.get_shape()[-1]), 128,
                                                         name="dense_6_conv_4_scale_3_1x1_3", filter_size=1)
        self.dense6_conv4_scale3_3 = self.conv_layer(self.dense6_conv4_scale3_1x1_3, 128, 164,
                                                     name="dense_6_conv_4_scale_3_3", filter_size=3)

        self.dense6_conv4_scale3 = tf.concat([self.dense6_conv4_scale3_2, self.dense6_conv4_scale3_3], axis=-1,
                                             name="dense_6_conv_4_scale_3")

        ## layer 5
        ### scale 1
        self.dense6_conv5_scale1_input = tf.concat(
            [self.dense6_conv1_scale1, self.dense6_conv2_scale1, self.dense6_conv3_scale1, self.dense6_conv4_scale1],
            axis=-1,
            name="dense_6_conv_5_scale_1_input")
        self.dense6_conv5_scale1_1x1 = self.conv_layer(self.dense6_conv5_scale1_input,
                                                       int(self.dense6_conv5_scale1_input.get_shape()[-1]), 96,
                                                       name="dense_6_conv_5_scale_1_1x1", filter_size=1)
        self.dense6_conv5_scale1 = self.conv_layer(self.dense6_conv5_scale1_1x1, 96, 128,
                                                   name="dense_6_conv_5_scale_1", filter_size=3)

        ### scale 2
        self.dense6_conv5_scale2_1x1_1 = self.conv_layer(self.dense6_conv5_scale1_input,
                                                         int(self.dense6_conv5_scale1_input.get_shape()[-1]), 128,
                                                         name="dense_6_conv_5_scale_2_1x1_1", filter_size=1)
        self.dense6_conv5_scale2_1 = self.conv_layer(self.dense6_conv5_scale2_1x1_1, 128, 164,
                                                     name="dense_6_conv_5_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense6_conv5_scale2_input = tf.concat(
            [self.dense6_conv1_scale2, self.dense6_conv2_scale2, self.dense6_conv3_scale2, self.dense6_conv4_scale2],
            axis=-1,
            name="dense_6_conv_5_scale_2_input")
        self.dense6_conv5_scale2_1x1_2 = self.conv_layer(self.dense6_conv5_scale2_input,
                                                         int(self.dense6_conv5_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_6_conv_5_scale_2_1x1_2", filter_size=1)
        self.dense6_conv5_scale2_2 = self.conv_layer(self.dense6_conv5_scale2_1x1_2, 128, 164,
                                                     name="dense_6_conv_5_scale_2_2", filter_size=3)

        self.dense6_conv5_scale2 = tf.concat([self.dense6_conv5_scale2_1, self.dense6_conv5_scale2_2], axis=-1,
                                             name="dense_6_conv_5_scale_2")

        ### scale 3
        self.dense6_conv5_scale3_1x1_2 = self.conv_layer(self.dense6_conv5_scale2_input,
                                                         int(self.dense6_conv5_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_6_conv_5_scale_3_1x1_2", filter_size=1)
        self.dense6_conv5_scale3_2 = self.conv_layer(self.dense6_conv5_scale3_1x1_2, 164, 198,
                                                     name="dense_6_conv_5_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense6_conv5_scale3_input = tf.concat(
            [self.dense6_conv1_scale3, self.dense6_conv2_scale3, self.dense6_conv3_scale3, self.dense6_conv4_scale3],
            axis=-1,
            name="dense_6_conv_5_scale_3_input")
        self.dense6_conv5_scale3_1x1_3 = self.conv_layer(self.dense6_conv5_scale3_input,
                                                         int(self.dense6_conv5_scale3_input.get_shape()[-1]), 164,
                                                         name="dense_6_conv_5_scale_3_1x1_3", filter_size=1)
        self.dense6_conv5_scale3_3 = self.conv_layer(self.dense6_conv5_scale3_1x1_3, 164, 198,
                                                     name="dense_6_conv_5_scale_3_3", filter_size=3)

        self.dense6_conv5_scale3 = tf.concat([self.dense6_conv5_scale3_2, self.dense6_conv5_scale3_3], axis=-1,
                                             name="dense_6_conv_5_scale_3")

        ## layer 6
        ### scale 1
        self.dense6_conv6_scale1_input = tf.concat(
            [self.dense6_conv1_scale1, self.dense6_conv2_scale1, self.dense6_conv3_scale1, self.dense6_conv4_scale1,
             self.dense6_conv5_scale1],
            axis=-1,
            name="dense_6_conv_6_scale_1_input")
        self.dense6_conv6_scale1_1x1 = self.conv_layer(self.dense6_conv6_scale1_input,
                                                       int(self.dense6_conv6_scale1_input.get_shape()[-1]), 128,
                                                       name="dense_6_conv_6_scale_1_1x1", filter_size=1)
        self.dense6_conv6_scale1 = self.conv_layer(self.dense6_conv6_scale1_1x1, 128, 164,
                                                   name="dense_6_conv_6_scale_1", filter_size=3)

        ### scale 2
        self.dense6_conv6_scale2_1x1_1 = self.conv_layer(self.dense6_conv6_scale1_input,
                                                         int(self.dense6_conv6_scale1_input.get_shape()[-1]), 164,
                                                         name="dense_6_conv_6_scale_2_1x1_1", filter_size=1)
        self.dense6_conv6_scale2_1 = self.conv_layer(self.dense6_conv6_scale2_1x1_1, 164, 198,
                                                     name="dense_6_conv_6_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense6_conv6_scale2_input = tf.concat(
            [self.dense6_conv1_scale2, self.dense6_conv2_scale2, self.dense6_conv3_scale2, self.dense6_conv4_scale2,
             self.dense6_conv5_scale2],
            axis=-1,
            name="dense_6_conv_6_scale_2_input")
        self.dense6_conv6_scale2_1x1_2 = self.conv_layer(self.dense6_conv6_scale2_input,
                                                         int(self.dense6_conv6_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_6_conv_6_scale_2_1x1_2", filter_size=1)
        self.dense6_conv6_scale2_2 = self.conv_layer(self.dense6_conv6_scale2_1x1_2, 164, 196,
                                                     name="dense_6_conv_6_scale_2_2", filter_size=3)

        self.dense6_conv6_scale2 = tf.concat([self.dense6_conv6_scale2_1, self.dense6_conv6_scale2_2], axis=-1,
                                             name="dense_6_conv_6_scale_2")

        ### scale 3
        self.dense6_conv6_scale3_1x1_2 = self.conv_layer(self.dense6_conv6_scale2_input,
                                                         int(self.dense6_conv6_scale2_input.get_shape()[-1]), 198,
                                                         name="dense_6_conv_6_scale_3_1x1_2", filter_size=1)
        self.dense6_conv6_scale3_2 = self.conv_layer(self.dense6_conv6_scale3_1x1_2, 198, 230,
                                                     name="dense_6_conv_6_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense6_conv6_scale3_input = tf.concat(
            [self.dense6_conv1_scale3, self.dense6_conv2_scale3, self.dense6_conv3_scale3, self.dense6_conv4_scale3,
             self.dense6_conv5_scale3],
            axis=-1,
            name="dense_6_conv_6_scale_3_input")
        self.dense6_conv6_scale3_1x1_3 = self.conv_layer(self.dense6_conv6_scale3_input,
                                                         int(self.dense6_conv6_scale3_input.get_shape()[-1]), 198,
                                                         name="dense_6_conv_6_scale_3_1x1_3", filter_size=1)
        self.dense6_conv6_scale3_3 = self.conv_layer(self.dense6_conv6_scale3_1x1_3, 198, 230,
                                                     name="dense_6_conv_6_scale_3_3", filter_size=3)

        self.dense6_conv6_scale3 = tf.concat([self.dense6_conv6_scale3_2, self.dense6_conv6_scale3_3], axis=-1,
                                             name="dense_6_conv_6_scale_3")

        # m output
        self.pool_m1 = self.max_pool(self.dense6_conv6_scale1, "pool_m_1")
        self.pool_m2 = self.max_pool(self.dense6_conv6_scale2, "pool_m_2")
        self.pool_m3 = self.max_pool(self.dense6_conv6_scale3, "pool_m_3")

        self.fc1_m1 = self.fc_layer(self.pool_m1, np.prod([int(x) for x in self.pool_m1.get_shape()[1:]]), 512,
                                    "fc_1_m_1")
        self.relu1_m1 = tf.nn.relu(self.fc1_m1)
        if train_mode == True:
            self.relu1_m1 = tf.nn.dropout(self.relu1_m1, 0.7)

        self.fc1_m2 = self.fc_layer(self.pool_m2, np.prod([int(x) for x in self.pool_m2.get_shape()[1:]]), 512,
                                    "fc_1_m_2")
        self.relu1_m2 = tf.nn.relu(self.fc1_m2)
        if train_mode == True:
            self.relu1_m2 = tf.nn.dropout(self.relu1_m2, 0.7)

        self.fc1_m3 = self.fc_layer(self.pool_m3, np.prod([int(x) for x in self.pool_m3.get_shape()[1:]]), 512,
                                    "fc_1_m_3")
        self.relu1_m3 = tf.nn.relu(self.fc1_m3)
        if train_mode == True:
            self.relu1_m3 = tf.nn.dropout(self.relu1_m3, 0.7)

        self.concat_m = tf.concat([self.relu1_m1, self.relu1_m2, self.relu1_m3], axis=-1)
        self.fc2_m = self.fc_layer(self.concat_m, 512 * 3, 1024, "fc_2_m")
        self.relu2_m = tf.nn.relu(self.fc2_m)
        if train_mode == True:
            self.relu2_m = tf.nn.dropout(self.relu2_m, 0.5)

        self.fc3_m = self.fc_layer(self.relu2_m, 1024, 1024, "fc_3_m")
        self.relu3_m = tf.nn.relu(self.fc3_m)
        if train_mode == True:
            self.relu3_m = tf.nn.dropout(self.relu3_m, 0.3)

        self.fc4_m = self.fc_layer(self.relu3_m, 1024, M_shape, "fc_4_m")
        self.m_output = tf.identity(self.fc4_m, name="m_out_put")

        # [finger I]
        # dense block 7
        ## layer 1
        ### scale 1
        self.dense7_conv1_scale1 = self.conv_layer(self.tran3_pool1, 24, 32, "dense_7_conv_1_scale_1", filter_size=3)
        ### scale 2
        self.dense7_conv1_scale2 = self.conv_layer(self.tran3_pool2, 32, 48, "dense_7_conv_1_scale_2", filter_size=3)
        ### scale 3
        self.dense7_conv1_scale3 = self.conv_layer(self.tran3_pool3, 48, 64, "dense_7_conv_1_scale_3", filter_size=3)

        ## layer 2
        ### scale 1
        self.dense7_conv2_scale1 = self.conv_layer(self.dense7_conv1_scale1, 32, 48, "dense_7_conv_2_scale_1",
                                                   filter_size=3)
        ### scale 2
        self.dense7_conv2_scale2_1 = self.conv_layer(self.dense7_conv1_scale1, 32, 48, "dense_7_conv_2_scale_2_1",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense7_conv2_scale2_2 = self.conv_layer(self.dense7_conv1_scale2, 48, 64, "dense_7_conv_2_scale_2_2",
                                                     filter_size=3)
        self.dense7_conv2_scale2 = tf.concat([self.dense7_conv2_scale2_1, self.dense7_conv2_scale2_2], axis=-1,
                                             name="dense_7_conv_2_scale_2")
        ### scale 3
        self.dense7_conv2_scale3_2 = self.conv_layer(self.dense7_conv1_scale2, 48, 64, "dense_7_conv_2_scale_3_2",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense7_conv2_scale3_3 = self.conv_layer(self.dense7_conv1_scale3, 64, 96, "dense_7_conv_2_scale_3_3",
                                                     filter_size=3)
        self.dense7_conv2_scale3 = tf.concat([self.dense7_conv2_scale3_2, self.dense7_conv2_scale3_3], axis=-1,
                                             name="dense_7_conv_2_scale_3")

        ## layer 3
        ### scale 1
        self.dense7_conv3_scale1_input = tf.concat([self.dense7_conv1_scale1, self.dense7_conv2_scale1], axis=-1,
                                                   name="dense_7_conv_3_scale_1_input")
        self.dense7_conv3_scale1_1x1 = self.conv_layer(self.dense7_conv3_scale1_input,
                                                       int(self.dense7_conv3_scale1_input.get_shape()[-1]), 48,
                                                       name="dense_7_conv_3_scale_1_1x1", filter_size=1)
        self.dense7_conv3_scale1 = self.conv_layer(self.dense7_conv3_scale1_1x1, 48, 64,
                                                   name="dense_7_conv_3_scale_1", filter_size=3)

        ### scale 2
        self.dense7_conv3_scale2_1x1_1 = self.conv_layer(self.dense7_conv3_scale1_input,
                                                         int(self.dense7_conv3_scale1_input.get_shape()[-1]), 64,
                                                         name="dense_7_conv_3_scale_2_1x1_1", filter_size=1)
        self.dense7_conv3_scale2_1 = self.conv_layer(self.dense7_conv3_scale2_1x1_1, 64, 96,
                                                     name="dense_7_conv_3_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense7_conv3_scale2_input = tf.concat([self.dense7_conv1_scale2, self.dense7_conv2_scale2], axis=-1,
                                                   name="dense_7_conv_3_scale_2_input")
        self.dense7_conv3_scale2_1x1_2 = self.conv_layer(self.dense7_conv3_scale2_input,
                                                         int(self.dense7_conv3_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_7_conv_3_scale_2_1x1_2", filter_size=1)
        self.dense7_conv3_scale2_2 = self.conv_layer(self.dense7_conv3_scale2_1x1_2, 64, 96,
                                                     name="dense_7_conv_3_scale_2_2", filter_size=3)

        self.dense7_conv3_scale2 = tf.concat([self.dense7_conv3_scale2_1, self.dense7_conv3_scale2_2], axis=-1,
                                             name="dense_7_conv_3_scale_2")

        ### scale 3
        self.dense7_conv3_scale3_1x1_2 = self.conv_layer(self.dense7_conv3_scale2_input,
                                                         int(self.dense7_conv3_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_7_conv_3_scale_3_1x1_2", filter_size=1)
        self.dense7_conv3_scale3_2 = self.conv_layer(self.dense7_conv3_scale3_1x1_2, 96, 128,
                                                     name="dense_7_conv_3_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense7_conv3_scale3_input = tf.concat([self.dense7_conv1_scale3, self.dense7_conv2_scale3], axis=-1,
                                                   name="dense_7_conv_3_scale_3_input")
        self.dense7_conv3_scale3_1x1_3 = self.conv_layer(self.dense7_conv3_scale3_input,
                                                         int(self.dense7_conv3_scale3_input.get_shape()[-1]), 96,
                                                         name="dense_7_conv_3_scale_3_1x1_3", filter_size=1)
        self.dense7_conv3_scale3_3 = self.conv_layer(self.dense7_conv3_scale3_1x1_3, 96, 128,
                                                     name="dense_7_conv_3_scale_3_3", filter_size=3)

        self.dense7_conv3_scale3 = tf.concat([self.dense7_conv3_scale3_2, self.dense7_conv3_scale3_3], axis=-1,
                                             name="dense_7_conv_3_scale_3")

        ## layer 4
        ### scale 1
        self.dense7_conv4_scale1_input = tf.concat(
            [self.dense7_conv1_scale1, self.dense7_conv2_scale1, self.dense7_conv3_scale1], axis=-1,
            name="dense_7_conv_4_scale_1_input")
        self.dense7_conv4_scale1_1x1 = self.conv_layer(self.dense7_conv4_scale1_input,
                                                       int(self.dense7_conv4_scale1_input.get_shape()[-1]), 64,
                                                       name="dense_7_conv_4_scale_1_1x1", filter_size=1)
        self.dense7_conv4_scale1 = self.conv_layer(self.dense7_conv4_scale1_1x1, 64, 96,
                                                   name="dense_7_conv_4_scale_1", filter_size=3)

        ### scale 2
        self.dense7_conv4_scale2_1x1_1 = self.conv_layer(self.dense7_conv4_scale1_input,
                                                         int(self.dense7_conv4_scale1_input.get_shape()[-1]), 96,
                                                         name="dense_7_conv_4_scale_2_1x1_1", filter_size=1)
        self.dense7_conv4_scale2_1 = self.conv_layer(self.dense7_conv4_scale2_1x1_1, 96, 128,
                                                     name="dense_7_conv_4_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense7_conv4_scale2_input = tf.concat(
            [self.dense7_conv1_scale2, self.dense7_conv2_scale2, self.dense7_conv3_scale2], axis=-1,
            name="dense_7_conv_4_scale_2_input")
        self.dense7_conv4_scale2_1x1_2 = self.conv_layer(self.dense7_conv4_scale2_input,
                                                         int(self.dense7_conv4_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_7_conv_4_scale_2_1x1_2", filter_size=1)
        self.dense7_conv4_scale2_2 = self.conv_layer(self.dense7_conv4_scale2_1x1_2, 96, 128,
                                                     name="dense_7_conv_4_scale_2_2", filter_size=3)

        self.dense7_conv4_scale2 = tf.concat([self.dense7_conv4_scale2_1, self.dense7_conv4_scale2_2], axis=-1,
                                             name="dense_7_conv_4_scale_2")

        ### scale 3
        self.dense7_conv4_scale3_1x1_2 = self.conv_layer(self.dense7_conv4_scale2_input,
                                                         int(self.dense7_conv4_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_7_conv_4_scale_3_1x1_2", filter_size=1)
        self.dense7_conv4_scale3_2 = self.conv_layer(self.dense7_conv4_scale3_1x1_2, 128, 164,
                                                     name="dense_7_conv_4_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense7_conv4_scale3_input = tf.concat(
            [self.dense7_conv1_scale3, self.dense7_conv2_scale3, self.dense7_conv3_scale3], axis=-1,
            name="dense_7_conv_4_scale_3_input")
        self.dense7_conv4_scale3_1x1_3 = self.conv_layer(self.dense7_conv4_scale3_input,
                                                         int(self.dense7_conv4_scale3_input.get_shape()[-1]), 128,
                                                         name="dense_7_conv_4_scale_3_1x1_3", filter_size=1)
        self.dense7_conv4_scale3_3 = self.conv_layer(self.dense7_conv4_scale3_1x1_3, 128, 164,
                                                     name="dense_7_conv_4_scale_3_3", filter_size=3)

        self.dense7_conv4_scale3 = tf.concat([self.dense7_conv4_scale3_2, self.dense7_conv4_scale3_3], axis=-1,
                                             name="dense_7_conv_4_scale_3")

        ## layer 5
        ### scale 1
        self.dense7_conv5_scale1_input = tf.concat(
            [self.dense7_conv1_scale1, self.dense7_conv2_scale1, self.dense7_conv3_scale1, self.dense7_conv4_scale1],
            axis=-1,
            name="dense_7_conv_5_scale_1_input")
        self.dense7_conv5_scale1_1x1 = self.conv_layer(self.dense7_conv5_scale1_input,
                                                       int(self.dense7_conv5_scale1_input.get_shape()[-1]), 96,
                                                       name="dense_7_conv_5_scale_1_1x1", filter_size=1)
        self.dense7_conv5_scale1 = self.conv_layer(self.dense7_conv5_scale1_1x1, 96, 128,
                                                   name="dense_7_conv_5_scale_1", filter_size=3)

        ### scale 2
        self.dense7_conv5_scale2_1x1_1 = self.conv_layer(self.dense7_conv5_scale1_input,
                                                         int(self.dense7_conv5_scale1_input.get_shape()[-1]), 128,
                                                         name="dense_7_conv_5_scale_2_1x1_1", filter_size=1)
        self.dense7_conv5_scale2_1 = self.conv_layer(self.dense7_conv5_scale2_1x1_1, 128, 164,
                                                     name="dense_7_conv_5_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense7_conv5_scale2_input = tf.concat(
            [self.dense7_conv1_scale2, self.dense7_conv2_scale2, self.dense7_conv3_scale2, self.dense7_conv4_scale2],
            axis=-1,
            name="dense_7_conv_5_scale_2_input")
        self.dense7_conv5_scale2_1x1_2 = self.conv_layer(self.dense7_conv5_scale2_input,
                                                         int(self.dense7_conv5_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_7_conv_5_scale_2_1x1_2", filter_size=1)
        self.dense7_conv5_scale2_2 = self.conv_layer(self.dense7_conv5_scale2_1x1_2, 128, 164,
                                                     name="dense_7_conv_5_scale_2_2", filter_size=3)

        self.dense7_conv5_scale2 = tf.concat([self.dense7_conv5_scale2_1, self.dense7_conv5_scale2_2], axis=-1,
                                             name="dense_7_conv_5_scale_2")

        ### scale 3
        self.dense7_conv5_scale3_1x1_2 = self.conv_layer(self.dense7_conv5_scale2_input,
                                                         int(self.dense7_conv5_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_7_conv_5_scale_3_1x1_2", filter_size=1)
        self.dense7_conv5_scale3_2 = self.conv_layer(self.dense7_conv5_scale3_1x1_2, 164, 198,
                                                     name="dense_7_conv_5_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense7_conv5_scale3_input = tf.concat(
            [self.dense7_conv1_scale3, self.dense7_conv2_scale3, self.dense7_conv3_scale3, self.dense7_conv4_scale3],
            axis=-1,
            name="dense_7_conv_5_scale_3_input")
        self.dense7_conv5_scale3_1x1_3 = self.conv_layer(self.dense7_conv5_scale3_input,
                                                         int(self.dense7_conv5_scale3_input.get_shape()[-1]), 164,
                                                         name="dense_7_conv_5_scale_3_1x1_3", filter_size=1)
        self.dense7_conv5_scale3_3 = self.conv_layer(self.dense7_conv5_scale3_1x1_3, 164, 198,
                                                     name="dense_7_conv_5_scale_3_3", filter_size=3)

        self.dense7_conv5_scale3 = tf.concat([self.dense7_conv5_scale3_2, self.dense7_conv5_scale3_3], axis=-1,
                                             name="dense_7_conv_5_scale_3")

        ## layer 6
        ### scale 1
        self.dense7_conv6_scale1_input = tf.concat(
            [self.dense7_conv1_scale1, self.dense7_conv2_scale1, self.dense7_conv3_scale1, self.dense7_conv4_scale1,
             self.dense7_conv5_scale1],
            axis=-1,
            name="dense_7_conv_6_scale_1_input")
        self.dense7_conv6_scale1_1x1 = self.conv_layer(self.dense7_conv6_scale1_input,
                                                       int(self.dense7_conv6_scale1_input.get_shape()[-1]), 128,
                                                       name="dense_7_conv_6_scale_1_1x1", filter_size=1)
        self.dense7_conv6_scale1 = self.conv_layer(self.dense7_conv6_scale1_1x1, 128, 164,
                                                   name="dense_7_conv_6_scale_1", filter_size=3)

        ### scale 2
        self.dense7_conv6_scale2_1x1_1 = self.conv_layer(self.dense7_conv6_scale1_input,
                                                         int(self.dense7_conv6_scale1_input.get_shape()[-1]), 164,
                                                         name="dense_7_conv_6_scale_2_1x1_1", filter_size=1)
        self.dense7_conv6_scale2_1 = self.conv_layer(self.dense7_conv6_scale2_1x1_1, 164, 198,
                                                     name="dense_7_conv_6_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense7_conv6_scale2_input = tf.concat(
            [self.dense7_conv1_scale2, self.dense7_conv2_scale2, self.dense7_conv3_scale2, self.dense7_conv4_scale2,
             self.dense7_conv5_scale2],
            axis=-1,
            name="dense_7_conv_6_scale_2_input")
        self.dense7_conv6_scale2_1x1_2 = self.conv_layer(self.dense7_conv6_scale2_input,
                                                         int(self.dense7_conv6_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_7_conv_6_scale_2_1x1_2", filter_size=1)
        self.dense7_conv6_scale2_2 = self.conv_layer(self.dense7_conv6_scale2_1x1_2, 164, 196,
                                                     name="dense_7_conv_6_scale_2_2", filter_size=3)

        self.dense7_conv6_scale2 = tf.concat([self.dense7_conv6_scale2_1, self.dense7_conv6_scale2_2], axis=-1,
                                             name="dense_7_conv_6_scale_2")

        ### scale 3
        self.dense7_conv6_scale3_1x1_2 = self.conv_layer(self.dense7_conv6_scale2_input,
                                                         int(self.dense7_conv6_scale2_input.get_shape()[-1]), 198,
                                                         name="dense_7_conv_6_scale_3_1x1_2", filter_size=1)
        self.dense7_conv6_scale3_2 = self.conv_layer(self.dense7_conv6_scale3_1x1_2, 198, 230,
                                                     name="dense_7_conv_6_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense7_conv6_scale3_input = tf.concat(
            [self.dense7_conv1_scale3, self.dense7_conv2_scale3, self.dense7_conv3_scale3, self.dense7_conv4_scale3,
             self.dense7_conv5_scale3],
            axis=-1,
            name="dense_7_conv_6_scale_3_input")
        self.dense7_conv6_scale3_1x1_3 = self.conv_layer(self.dense7_conv6_scale3_input,
                                                         int(self.dense7_conv6_scale3_input.get_shape()[-1]), 198,
                                                         name="dense_7_conv_6_scale_3_1x1_3", filter_size=1)
        self.dense7_conv6_scale3_3 = self.conv_layer(self.dense7_conv6_scale3_1x1_3, 198, 230,
                                                     name="dense_7_conv_6_scale_3_3", filter_size=3)

        self.dense7_conv6_scale3 = tf.concat([self.dense7_conv6_scale3_2, self.dense7_conv6_scale3_3], axis=-1,
                                             name="dense_7_conv_6_scale_3")

        # r output
        self.pool_i1 = self.max_pool(self.dense7_conv6_scale1, "pool_i_1")
        self.pool_i2 = self.max_pool(self.dense7_conv6_scale2, "pool_i_2")
        self.pool_i3 = self.max_pool(self.dense7_conv6_scale3, "pool_i_3")

        self.fc1_i1 = self.fc_layer(self.pool_i1, np.prod([int(x) for x in self.pool_i1.get_shape()[1:]]), 512,
                                    "fc_1_i_1")
        self.relu1_i1 = tf.nn.relu(self.fc1_i1)
        if train_mode == True:
            self.relu1_i1 = tf.nn.dropout(self.relu1_i1, 0.7)

        self.fc1_i2 = self.fc_layer(self.pool_i2, np.prod([int(x) for x in self.pool_i2.get_shape()[1:]]), 512,
                                    "fc_1_i_2")
        self.relu1_i2 = tf.nn.relu(self.fc1_i2)
        if train_mode == True:
            self.relu1_i2 = tf.nn.dropout(self.relu1_i2, 0.7)

        self.fc1_i3 = self.fc_layer(self.pool_i3, np.prod([int(x) for x in self.pool_i3.get_shape()[1:]]), 512,
                                    "fc_1_i_3")
        self.relu1_i3 = tf.nn.relu(self.fc1_i3)
        if train_mode == True:
            self.relu1_i3 = tf.nn.dropout(self.relu1_i3, 0.7)

        self.concat_i = tf.concat([self.relu1_i1, self.relu1_i2, self.relu1_i3], axis=-1)
        self.fc2_i = self.fc_layer(self.concat_i, 512 * 3, 1024, "fc_2_i")
        self.relu2_i = tf.nn.relu(self.fc2_i)
        if train_mode == True:
            self.relu2_i = tf.nn.dropout(self.relu2_i, 0.5)

        self.fc3_i = self.fc_layer(self.relu2_i, 1024, 1024, "fc_3_i")
        self.relu3_i = tf.nn.relu(self.fc3_i)
        if train_mode == True:
            self.relu3_i = tf.nn.dropout(self.relu3_i, 0.3)

        self.fc4_i = self.fc_layer(self.relu3_i, 1024, I_shape, "fc_4_i")
        self.i_output = tf.identity(self.fc4_i, name="i_out_put")

        # [finger T]
        # dense block 8
        ## layer 1
        ### scale 1
        self.dense8_conv1_scale1 = self.conv_layer(self.tran1_pool1, 16, 24, "dense_8_conv_1_scale_1", filter_size=3)
        ### scale 2
        self.dense8_conv1_scale2 = self.conv_layer(self.tran1_pool2, 24, 32, "dense_8_conv_1_scale_2", filter_size=3)
        ### scale 3
        self.dense8_conv1_scale3 = self.conv_layer(self.tran1_pool3, 32, 48, "dense_8_conv_1_scale_3", filter_size=3)

        ## layer 2
        ### scale 1
        self.dense8_conv2_scale1 = self.conv_layer(self.dense8_conv1_scale1, 24, 32, "dense_8_conv_2_scale_1",
                                                   filter_size=3)
        ### scale 2
        self.dense8_conv2_scale2_1 = self.conv_layer(self.dense8_conv1_scale1, 24, 32, "dense_8_conv_2_scale_2_1",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense8_conv2_scale2_2 = self.conv_layer(self.dense8_conv1_scale2, 32, 48, "dense_8_conv_2_scale_2_2",
                                                     filter_size=3)
        self.dense8_conv2_scale2 = tf.concat([self.dense8_conv2_scale2_1, self.dense8_conv2_scale2_2], axis=-1,
                                             name="dense_8_conv_2_scale_2")
        ### scale 3
        self.dense8_conv2_scale3_2 = self.conv_layer(self.dense8_conv1_scale2, 32, 48, "dense_8_conv_2_scale_3_2",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense8_conv2_scale3_3 = self.conv_layer(self.dense8_conv1_scale3, 48, 64, "dense_8_conv_2_scale_3_3",
                                                     filter_size=3)
        self.dense8_conv2_scale3 = tf.concat([self.dense8_conv2_scale3_2, self.dense8_conv2_scale3_3], axis=-1,
                                             name="dense_8_conv_2_scale_3")

        ## layer 3
        ### scale 1
        self.dense8_conv3_scale1_input = tf.concat([self.dense8_conv1_scale1, self.dense8_conv2_scale1], axis=-1,
                                                   name="dense_8_conv_3_scale_1_input")
        self.dense8_conv3_scale1_1x1 = self.conv_layer(self.dense8_conv3_scale1_input,
                                                       int(self.dense8_conv3_scale1_input.get_shape()[-1]), 32,
                                                       name="dense_8_conv_3_scale_1_1x1", filter_size=1)
        self.dense8_conv3_scale1 = self.conv_layer(self.dense8_conv3_scale1_1x1, 32, 48,
                                                   name="dense_8_conv_3_scale_1", filter_size=3)

        ### scale 2
        self.dense8_conv3_scale2_1x1_1 = self.conv_layer(self.dense8_conv3_scale1_input,
                                                         int(self.dense8_conv3_scale1_input.get_shape()[-1]), 48,
                                                         name="dense_8_conv_3_scale_2_1x1_1", filter_size=1)
        self.dense8_conv3_scale2_1 = self.conv_layer(self.dense8_conv3_scale2_1x1_1, 48, 64,
                                                     name="dense_8_conv_3_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense8_conv3_scale2_input = tf.concat([self.dense8_conv1_scale2, self.dense8_conv2_scale2], axis=-1,
                                                   name="dense_8_conv_3_scale_2_input")
        self.dense8_conv3_scale2_1x1_2 = self.conv_layer(self.dense8_conv3_scale2_input,
                                                         int(self.dense8_conv3_scale2_input.get_shape()[-1]), 48,
                                                         name="dense_8_conv_3_scale_2_1x1_2", filter_size=1)
        self.dense8_conv3_scale2_2 = self.conv_layer(self.dense8_conv3_scale2_1x1_2, 48, 64,
                                                     name="dense_8_conv_3_scale_2_2", filter_size=3)

        self.dense8_conv3_scale2 = tf.concat([self.dense8_conv3_scale2_1, self.dense8_conv3_scale2_2], axis=-1,
                                             name="dense_8_conv_3_scale_2")

        ### scale 3
        self.dense8_conv3_scale3_1x1_2 = self.conv_layer(self.dense8_conv3_scale2_input,
                                                         int(self.dense8_conv3_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_8_conv_3_scale_3_1x1_2", filter_size=1)
        self.dense8_conv3_scale3_2 = self.conv_layer(self.dense8_conv3_scale3_1x1_2, 64, 96,
                                                     name="dense_8_conv_3_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense8_conv3_scale3_input = tf.concat([self.dense8_conv1_scale3, self.dense8_conv2_scale3], axis=-1,
                                                   name="dense_8_conv_3_scale_3_input")
        self.dense8_conv3_scale3_1x1_3 = self.conv_layer(self.dense8_conv3_scale3_input,
                                                         int(self.dense8_conv3_scale3_input.get_shape()[-1]), 64,
                                                         name="dense_8_conv_3_scale_3_1x1_3", filter_size=1)
        self.dense8_conv3_scale3_3 = self.conv_layer(self.dense8_conv3_scale3_1x1_3, 64, 96,
                                                     name="dense_8_conv_3_scale_3_3", filter_size=3)

        self.dense8_conv3_scale3 = tf.concat([self.dense8_conv3_scale3_2, self.dense8_conv3_scale3_3], axis=-1,
                                             name="dense_8_conv_3_scale_3")

        ## layer 4
        ### scale 1
        self.dense8_conv4_scale1_input = tf.concat(
            [self.dense8_conv1_scale1, self.dense8_conv2_scale1, self.dense8_conv3_scale1], axis=-1,
            name="dense_8_conv_4_scale_1_input")
        self.dense8_conv4_scale1_1x1 = self.conv_layer(self.dense8_conv4_scale1_input,
                                                       int(self.dense8_conv4_scale1_input.get_shape()[-1]), 48,
                                                       name="dense_8_conv_4_scale_1_1x1", filter_size=1)
        self.dense8_conv4_scale1 = self.conv_layer(self.dense8_conv4_scale1_1x1, 48, 64,
                                                   name="dense_8_conv_4_scale_1", filter_size=3)

        ### scale 2
        self.dense8_conv4_scale2_1x1_1 = self.conv_layer(self.dense8_conv4_scale1_input,
                                                         int(self.dense8_conv4_scale1_input.get_shape()[-1]), 64,
                                                         name="dense_8_conv_4_scale_2_1x1_1", filter_size=1)
        self.dense8_conv4_scale2_1 = self.conv_layer(self.dense8_conv4_scale2_1x1_1, 64, 96,
                                                     name="dense_8_conv_4_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense8_conv4_scale2_input = tf.concat(
            [self.dense8_conv1_scale2, self.dense8_conv2_scale2, self.dense8_conv3_scale2], axis=-1,
            name="dense_8_conv_4_scale_2_input")
        self.dense8_conv4_scale2_1x1_2 = self.conv_layer(self.dense8_conv4_scale2_input,
                                                         int(self.dense8_conv4_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_8_conv_4_scale_2_1x1_2", filter_size=1)
        self.dense8_conv4_scale2_2 = self.conv_layer(self.dense8_conv4_scale2_1x1_2, 64, 96,
                                                     name="dense_8_conv_4_scale_2_2", filter_size=3)

        self.dense8_conv4_scale2 = tf.concat([self.dense8_conv4_scale2_1, self.dense8_conv4_scale2_2], axis=-1,
                                             name="dense_8_conv_4_scale_2")

        ### scale 3
        self.dense8_conv4_scale3_1x1_2 = self.conv_layer(self.dense8_conv4_scale2_input,
                                                         int(self.dense8_conv4_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_8_conv_4_scale_3_1x1_2", filter_size=1)
        self.dense8_conv4_scale3_2 = self.conv_layer(self.dense8_conv4_scale3_1x1_2, 96, 128,
                                                     name="dense_8_conv_4_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense8_conv4_scale3_input = tf.concat(
            [self.dense8_conv1_scale3, self.dense8_conv2_scale3, self.dense8_conv3_scale3], axis=-1,
            name="dense_8_conv_4_scale_3_input")
        self.dense8_conv4_scale3_1x1_3 = self.conv_layer(self.dense8_conv4_scale3_input,
                                                         int(self.dense8_conv4_scale3_input.get_shape()[-1]), 96,
                                                         name="dense_8_conv_4_scale_3_1x1_3", filter_size=1)
        self.dense8_conv4_scale3_3 = self.conv_layer(self.dense8_conv4_scale3_1x1_3, 96, 128,
                                                     name="dense_8_conv_4_scale_3_3", filter_size=3)

        self.dense8_conv4_scale3 = tf.concat([self.dense8_conv4_scale3_2, self.dense8_conv4_scale3_3], axis=-1,
                                             name="dense_8_conv_4_scale_3")

        # transition 4
        self.tran4_conv1 = self.conv_layer(self.dense8_conv4_scale1, int(self.dense8_conv4_scale1.get_shape()[-1]), 24,
                                           name="tran_4_conv_1", filter_size=1)
        self.tran4_pool1 = self.max_pool(self.tran4_conv1, name="tran_4_pool_1")

        self.tran4_conv2 = self.conv_layer(self.dense8_conv4_scale2, int(self.dense8_conv4_scale2.get_shape()[-1]), 32,
                                           name="tran_4_conv_2", filter_size=1)
        self.tran4_pool2 = self.max_pool(self.tran4_conv2, name="tran_4_pool_2")

        self.tran4_conv3 = self.conv_layer(self.dense8_conv4_scale3, int(self.dense8_conv4_scale3.get_shape()[-1]), 48,
                                           name="tran_4_conv_3", filter_size=1)
        self.tran4_pool3 = self.max_pool(self.tran4_conv3, name="tran_4_pool_3")

        # [finger M]
        # dense block 6
        ## layer 1
        ### scale 1
        self.dense9_conv1_scale1 = self.conv_layer(self.tran4_pool1, 24, 32, "dense_9_conv_1_scale_1", filter_size=3)
        ### scale 2
        self.dense9_conv1_scale2 = self.conv_layer(self.tran4_pool2, 32, 48, "dense_9_conv_1_scale_2", filter_size=3)
        ### scale 3
        self.dense9_conv1_scale3 = self.conv_layer(self.tran4_pool3, 48, 64, "dense_9_conv_1_scale_3", filter_size=3)

        ## layer 2
        ### scale 1
        self.dense9_conv2_scale1 = self.conv_layer(self.dense9_conv1_scale1, 32, 48, "dense_9_conv_2_scale_1",
                                                   filter_size=3)
        ### scale 2
        self.dense9_conv2_scale2_1 = self.conv_layer(self.dense9_conv1_scale1, 32, 48, "dense_9_conv_2_scale_2_1",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense9_conv2_scale2_2 = self.conv_layer(self.dense9_conv1_scale2, 48, 64, "dense_9_conv_2_scale_2_2",
                                                     filter_size=3)
        self.dense9_conv2_scale2 = tf.concat([self.dense9_conv2_scale2_1, self.dense9_conv2_scale2_2], axis=-1,
                                             name="dense_9_conv_2_scale_2")
        ### scale 3
        self.dense9_conv2_scale3_2 = self.conv_layer(self.dense9_conv1_scale2, 48, 64, "dense_9_conv_2_scale_3_2",
                                                     filter_size=3, stride=[1, 2, 2, 1])
        self.dense9_conv2_scale3_3 = self.conv_layer(self.dense9_conv1_scale3, 64, 96, "dense_9_conv_2_scale_3_3",
                                                     filter_size=3)
        self.dense9_conv2_scale3 = tf.concat([self.dense9_conv2_scale3_2, self.dense9_conv2_scale3_3], axis=-1,
                                             name="dense_9_conv_2_scale_3")

        ## layer 3
        ### scale 1
        self.dense9_conv3_scale1_input = tf.concat([self.dense9_conv1_scale1, self.dense9_conv2_scale1], axis=-1,
                                                   name="dense_9_conv_3_scale_1_input")
        self.dense9_conv3_scale1_1x1 = self.conv_layer(self.dense9_conv3_scale1_input,
                                                       int(self.dense9_conv3_scale1_input.get_shape()[-1]), 48,
                                                       name="dense_9_conv_3_scale_1_1x1", filter_size=1)
        self.dense9_conv3_scale1 = self.conv_layer(self.dense9_conv3_scale1_1x1, 48, 64,
                                                   name="dense_9_conv_3_scale_1", filter_size=3)

        ### scale 2
        self.dense9_conv3_scale2_1x1_1 = self.conv_layer(self.dense9_conv3_scale1_input,
                                                         int(self.dense9_conv3_scale1_input.get_shape()[-1]), 64,
                                                         name="dense_9_conv_3_scale_2_1x1_1", filter_size=1)
        self.dense9_conv3_scale2_1 = self.conv_layer(self.dense9_conv3_scale2_1x1_1, 64, 96,
                                                     name="dense_9_conv_3_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense9_conv3_scale2_input = tf.concat([self.dense9_conv1_scale2, self.dense9_conv2_scale2], axis=-1,
                                                   name="dense_9_conv_3_scale_2_input")
        self.dense9_conv3_scale2_1x1_2 = self.conv_layer(self.dense9_conv3_scale2_input,
                                                         int(self.dense9_conv3_scale2_input.get_shape()[-1]), 64,
                                                         name="dense_9_conv_3_scale_2_1x1_2", filter_size=1)
        self.dense9_conv3_scale2_2 = self.conv_layer(self.dense9_conv3_scale2_1x1_2, 64, 96,
                                                     name="dense_9_conv_3_scale_2_2", filter_size=3)

        self.dense9_conv3_scale2 = tf.concat([self.dense9_conv3_scale2_1, self.dense9_conv3_scale2_2], axis=-1,
                                             name="dense_9_conv_3_scale_2")

        ### scale 3
        self.dense9_conv3_scale3_1x1_2 = self.conv_layer(self.dense9_conv3_scale2_input,
                                                         int(self.dense9_conv3_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_9_conv_3_scale_3_1x1_2", filter_size=1)
        self.dense9_conv3_scale3_2 = self.conv_layer(self.dense9_conv3_scale3_1x1_2, 96, 128,
                                                     name="dense_9_conv_3_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense9_conv3_scale3_input = tf.concat([self.dense9_conv1_scale3, self.dense9_conv2_scale3], axis=-1,
                                                   name="dense_9_conv_3_scale_3_input")
        self.dense9_conv3_scale3_1x1_3 = self.conv_layer(self.dense9_conv3_scale3_input,
                                                         int(self.dense9_conv3_scale3_input.get_shape()[-1]), 96,
                                                         name="dense_9_conv_3_scale_3_1x1_3", filter_size=1)
        self.dense9_conv3_scale3_3 = self.conv_layer(self.dense9_conv3_scale3_1x1_3, 96, 128,
                                                     name="dense_9_conv_3_scale_3_3", filter_size=3)

        self.dense9_conv3_scale3 = tf.concat([self.dense9_conv3_scale3_2, self.dense9_conv3_scale3_3], axis=-1,
                                             name="dense_9_conv_3_scale_3")

        ## layer 4
        ### scale 1
        self.dense9_conv4_scale1_input = tf.concat(
            [self.dense9_conv1_scale1, self.dense9_conv2_scale1, self.dense9_conv3_scale1], axis=-1,
            name="dense_9_conv_4_scale_1_input")
        self.dense9_conv4_scale1_1x1 = self.conv_layer(self.dense9_conv4_scale1_input,
                                                       int(self.dense9_conv4_scale1_input.get_shape()[-1]), 64,
                                                       name="dense_9_conv_4_scale_1_1x1", filter_size=1)
        self.dense9_conv4_scale1 = self.conv_layer(self.dense9_conv4_scale1_1x1, 64, 96,
                                                   name="dense_9_conv_4_scale_1", filter_size=3)

        ### scale 2
        self.dense9_conv4_scale2_1x1_1 = self.conv_layer(self.dense9_conv4_scale1_input,
                                                         int(self.dense9_conv4_scale1_input.get_shape()[-1]), 96,
                                                         name="dense_9_conv_4_scale_2_1x1_1", filter_size=1)
        self.dense9_conv4_scale2_1 = self.conv_layer(self.dense9_conv4_scale2_1x1_1, 96, 128,
                                                     name="dense_9_conv_4_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense9_conv4_scale2_input = tf.concat(
            [self.dense9_conv1_scale2, self.dense9_conv2_scale2, self.dense9_conv3_scale2], axis=-1,
            name="dense_9_conv_4_scale_2_input")
        self.dense9_conv4_scale2_1x1_2 = self.conv_layer(self.dense9_conv4_scale2_input,
                                                         int(self.dense9_conv4_scale2_input.get_shape()[-1]), 96,
                                                         name="dense_9_conv_4_scale_2_1x1_2", filter_size=1)
        self.dense9_conv4_scale2_2 = self.conv_layer(self.dense9_conv4_scale2_1x1_2, 96, 128,
                                                     name="dense_9_conv_4_scale_2_2", filter_size=3)

        self.dense9_conv4_scale2 = tf.concat([self.dense9_conv4_scale2_1, self.dense9_conv4_scale2_2], axis=-1,
                                             name="dense_9_conv_4_scale_2")

        ### scale 3
        self.dense9_conv4_scale3_1x1_2 = self.conv_layer(self.dense9_conv4_scale2_input,
                                                         int(self.dense9_conv4_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_9_conv_4_scale_3_1x1_2", filter_size=1)
        self.dense9_conv4_scale3_2 = self.conv_layer(self.dense9_conv4_scale3_1x1_2, 128, 164,
                                                     name="dense_9_conv_4_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense9_conv4_scale3_input = tf.concat(
            [self.dense9_conv1_scale3, self.dense9_conv2_scale3, self.dense9_conv3_scale3], axis=-1,
            name="dense_9_conv_4_scale_3_input")
        self.dense9_conv4_scale3_1x1_3 = self.conv_layer(self.dense9_conv4_scale3_input,
                                                         int(self.dense9_conv4_scale3_input.get_shape()[-1]), 128,
                                                         name="dense_9_conv_4_scale_3_1x1_3", filter_size=1)
        self.dense9_conv4_scale3_3 = self.conv_layer(self.dense9_conv4_scale3_1x1_3, 128, 164,
                                                     name="dense_9_conv_4_scale_3_3", filter_size=3)

        self.dense9_conv4_scale3 = tf.concat([self.dense9_conv4_scale3_2, self.dense9_conv4_scale3_3], axis=-1,
                                             name="dense_9_conv_4_scale_3")

        ## layer 5
        ### scale 1
        self.dense9_conv5_scale1_input = tf.concat(
            [self.dense9_conv1_scale1, self.dense9_conv2_scale1, self.dense9_conv3_scale1, self.dense9_conv4_scale1],
            axis=-1,
            name="dense_9_conv_5_scale_1_input")
        self.dense9_conv5_scale1_1x1 = self.conv_layer(self.dense9_conv5_scale1_input,
                                                       int(self.dense9_conv5_scale1_input.get_shape()[-1]), 96,
                                                       name="dense_9_conv_5_scale_1_1x1", filter_size=1)
        self.dense9_conv5_scale1 = self.conv_layer(self.dense9_conv5_scale1_1x1, 96, 128,
                                                   name="dense_9_conv_5_scale_1", filter_size=3)

        ### scale 2
        self.dense9_conv5_scale2_1x1_1 = self.conv_layer(self.dense9_conv5_scale1_input,
                                                         int(self.dense9_conv5_scale1_input.get_shape()[-1]), 128,
                                                         name="dense_9_conv_5_scale_2_1x1_1", filter_size=1)
        self.dense9_conv5_scale2_1 = self.conv_layer(self.dense9_conv5_scale2_1x1_1, 128, 164,
                                                     name="dense_9_conv_5_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense9_conv5_scale2_input = tf.concat(
            [self.dense9_conv1_scale2, self.dense9_conv2_scale2, self.dense9_conv3_scale2, self.dense9_conv4_scale2],
            axis=-1,
            name="dense_9_conv_5_scale_2_input")
        self.dense9_conv5_scale2_1x1_2 = self.conv_layer(self.dense9_conv5_scale2_input,
                                                         int(self.dense9_conv5_scale2_input.get_shape()[-1]), 128,
                                                         name="dense_9_conv_5_scale_2_1x1_2", filter_size=1)
        self.dense9_conv5_scale2_2 = self.conv_layer(self.dense9_conv5_scale2_1x1_2, 128, 164,
                                                     name="dense_9_conv_5_scale_2_2", filter_size=3)

        self.dense9_conv5_scale2 = tf.concat([self.dense9_conv5_scale2_1, self.dense9_conv5_scale2_2], axis=-1,
                                             name="dense_9_conv_5_scale_2")

        ### scale 3
        self.dense9_conv5_scale3_1x1_2 = self.conv_layer(self.dense9_conv5_scale2_input,
                                                         int(self.dense9_conv5_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_9_conv_5_scale_3_1x1_2", filter_size=1)
        self.dense9_conv5_scale3_2 = self.conv_layer(self.dense9_conv5_scale3_1x1_2, 164, 198,
                                                     name="dense_9_conv_5_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense9_conv5_scale3_input = tf.concat(
            [self.dense9_conv1_scale3, self.dense9_conv2_scale3, self.dense9_conv3_scale3, self.dense9_conv4_scale3],
            axis=-1,
            name="dense_9_conv_5_scale_3_input")
        self.dense9_conv5_scale3_1x1_3 = self.conv_layer(self.dense9_conv5_scale3_input,
                                                         int(self.dense9_conv5_scale3_input.get_shape()[-1]), 164,
                                                         name="dense_9_conv_5_scale_3_1x1_3", filter_size=1)
        self.dense9_conv5_scale3_3 = self.conv_layer(self.dense9_conv5_scale3_1x1_3, 164, 198,
                                                     name="dense_9_conv_5_scale_3_3", filter_size=3)

        self.dense9_conv5_scale3 = tf.concat([self.dense9_conv5_scale3_2, self.dense9_conv5_scale3_3], axis=-1,
                                             name="dense_9_conv_5_scale_3")

        ## layer 6
        ### scale 1
        self.dense9_conv6_scale1_input = tf.concat(
            [self.dense9_conv1_scale1, self.dense9_conv2_scale1, self.dense9_conv3_scale1, self.dense9_conv4_scale1,
             self.dense9_conv5_scale1],
            axis=-1,
            name="dense_9_conv_6_scale_1_input")
        self.dense9_conv6_scale1_1x1 = self.conv_layer(self.dense9_conv6_scale1_input,
                                                       int(self.dense9_conv6_scale1_input.get_shape()[-1]), 128,
                                                       name="dense_9_conv_6_scale_1_1x1", filter_size=1)
        self.dense9_conv6_scale1 = self.conv_layer(self.dense9_conv6_scale1_1x1, 128, 164,
                                                   name="dense_9_conv_6_scale_1", filter_size=3)

        ### scale 2
        self.dense9_conv6_scale2_1x1_1 = self.conv_layer(self.dense9_conv6_scale1_input,
                                                         int(self.dense9_conv6_scale1_input.get_shape()[-1]), 164,
                                                         name="dense_9_conv_6_scale_2_1x1_1", filter_size=1)
        self.dense9_conv6_scale2_1 = self.conv_layer(self.dense9_conv6_scale2_1x1_1, 164, 198,
                                                     name="dense_9_conv_6_scale_2_1", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense9_conv6_scale2_input = tf.concat(
            [self.dense9_conv1_scale2, self.dense9_conv2_scale2, self.dense9_conv3_scale2, self.dense9_conv4_scale2,
             self.dense9_conv5_scale2],
            axis=-1,
            name="dense_9_conv_6_scale_2_input")
        self.dense9_conv6_scale2_1x1_2 = self.conv_layer(self.dense9_conv6_scale2_input,
                                                         int(self.dense9_conv6_scale2_input.get_shape()[-1]), 164,
                                                         name="dense_9_conv_6_scale_2_1x1_2", filter_size=1)
        self.dense9_conv6_scale2_2 = self.conv_layer(self.dense9_conv6_scale2_1x1_2, 164, 196,
                                                     name="dense_9_conv_6_scale_2_2", filter_size=3)

        self.dense9_conv6_scale2 = tf.concat([self.dense9_conv6_scale2_1, self.dense9_conv6_scale2_2], axis=-1,
                                             name="dense_9_conv_6_scale_2")

        ### scale 3
        self.dense9_conv6_scale3_1x1_2 = self.conv_layer(self.dense9_conv6_scale2_input,
                                                         int(self.dense9_conv6_scale2_input.get_shape()[-1]), 198,
                                                         name="dense_9_conv_6_scale_3_1x1_2", filter_size=1)
        self.dense9_conv6_scale3_2 = self.conv_layer(self.dense9_conv6_scale3_1x1_2, 198, 230,
                                                     name="dense_9_conv_6_scale_3_2", filter_size=3,
                                                     stride=[1, 2, 2, 1])

        self.dense9_conv6_scale3_input = tf.concat(
            [self.dense9_conv1_scale3, self.dense9_conv2_scale3, self.dense9_conv3_scale3, self.dense9_conv4_scale3,
             self.dense9_conv5_scale3],
            axis=-1,
            name="dense_9_conv_6_scale_3_input")
        self.dense9_conv6_scale3_1x1_3 = self.conv_layer(self.dense9_conv6_scale3_input,
                                                         int(self.dense9_conv6_scale3_input.get_shape()[-1]), 198,
                                                         name="dense_9_conv_6_scale_3_1x1_3", filter_size=1)
        self.dense9_conv6_scale3_3 = self.conv_layer(self.dense9_conv6_scale3_1x1_3, 198, 230,
                                                     name="dense_9_conv_6_scale_3_3", filter_size=3)

        self.dense9_conv6_scale3 = tf.concat([self.dense9_conv6_scale3_2, self.dense9_conv6_scale3_3], axis=-1,
                                             name="dense_9_conv_6_scale_3")

        # m output
        self.pool_t1 = self.max_pool(self.dense9_conv6_scale1, "pool_t_1")
        self.pool_t2 = self.max_pool(self.dense9_conv6_scale2, "pool_t_2")
        self.pool_t3 = self.max_pool(self.dense9_conv6_scale3, "pool_t_3")

        self.fc1_t1 = self.fc_layer(self.pool_t1, np.prod([int(x) for x in self.pool_t1.get_shape()[1:]]), 512,
                                    "fc_1_t_1")
        self.relu1_t1 = tf.nn.relu(self.fc1_t1)
        if train_mode == True:
            self.relu1_t1 = tf.nn.dropout(self.relu1_t1, 0.7)

        self.fc1_t2 = self.fc_layer(self.pool_t2, np.prod([int(x) for x in self.pool_t2.get_shape()[1:]]), 512,
                                    "fc_1_t_2")
        self.relu1_t2 = tf.nn.relu(self.fc1_t2)
        if train_mode == True:
            self.relu1_t2 = tf.nn.dropout(self.relu1_t2, 0.7)

        self.fc1_t3 = self.fc_layer(self.pool_t3, np.prod([int(x) for x in self.pool_t3.get_shape()[1:]]), 512,
                                    "fc_1_t_3")
        self.relu1_t3 = tf.nn.relu(self.fc1_t3)
        if train_mode == True:
            self.relu1_t3 = tf.nn.dropout(self.relu1_t3, 0.7)

        self.concat_t = tf.concat([self.relu1_t1, self.relu1_t2, self.relu1_t3], axis=-1)
        self.fc2_t = self.fc_layer(self.concat_t, 512 * 3, 1024, "fc_2_t")
        self.relu2_t = tf.nn.relu(self.fc2_t)
        if train_mode == True:
            self.relu2_t = tf.nn.dropout(self.relu2_t, 0.5)

        self.fc3_t = self.fc_layer(self.relu2_t, 1024, 1024, "fc_3_t")
        self.relu3_t = tf.nn.relu(self.fc3_t)
        if train_mode == True:
            self.relu3_t = tf.nn.dropout(self.relu3_t, 0.3)

        self.fc4_t = self.fc_layer(self.relu3_t, 1024, T_shape, "fc_4_t")
        self.t_output = tf.identity(self.fc4_t, name="t_out_put")

        # whole hand
        self.fc1_ph1 = self.fc_layer(self.pool_p1, np.prod([int(x) for x in self.pool_p1.get_shape()[1:]]), 512,
                                    "fc_1_ph_1")
        self.relu1_ph1 = tf.nn.relu(self.fc1_ph1)
        if train_mode == True:
            self.relu1_ph1 = tf.nn.dropout(self.relu1_ph1, 0.7)

        self.fc1_ph2 = self.fc_layer(self.pool_p2, np.prod([int(x) for x in self.pool_p2.get_shape()[1:]]), 512,
                                    "fc_1_ph_2")
        self.relu1_ph2 = tf.nn.relu(self.fc1_ph2)
        if train_mode == True:
            self.relu1_ph2 = tf.nn.dropout(self.relu1_ph2, 0.7)

        self.fc1_ph3 = self.fc_layer(self.pool_p3, np.prod([int(x) for x in self.pool_p3.get_shape()[1:]]), 512,
                                    "fc_1_ph_3")
        self.relu1_ph3 = tf.nn.relu(self.fc1_ph3)
        if train_mode == True:
            self.relu1_ph3 = tf.nn.dropout(self.relu1_ph3, 0.7)

        self.concat_ph = tf.concat([self.relu1_ph1, self.relu1_ph2, self.relu1_ph3], axis=-1)
        self.fc2_ph = self.fc_layer(self.concat_ph, 512 * 3, 1024, "fc_2_ph")
        self.relu2_ph = tf.nn.relu(self.fc2_ph)
        if train_mode == True:
            self.relu2_ph = tf.nn.dropout(self.relu2_ph, 0.5)



        self.fc1_rh1 = self.fc_layer(self.pool_r1, np.prod([int(x) for x in self.pool_r1.get_shape()[1:]]), 512,
                                     "fc_1_rh_1")
        self.relu1_rh1 = tf.nn.relu(self.fc1_rh1)
        if train_mode == True:
            self.relu1_rh1 = tf.nn.dropout(self.relu1_rh1, 0.7)

        self.fc1_rh2 = self.fc_layer(self.pool_r2, np.prod([int(x) for x in self.pool_r2.get_shape()[1:]]), 512,
                                     "fc_1_rh_2")
        self.relu1_rh2 = tf.nn.relu(self.fc1_rh2)
        if train_mode == True:
            self.relu1_rh2 = tf.nn.dropout(self.relu1_rh2, 0.7)

        self.fc1_rh3 = self.fc_layer(self.pool_r3, np.prod([int(x) for x in self.pool_r3.get_shape()[1:]]), 512,
                                     "fc_1_rh_3")
        self.relu1_rh3 = tf.nn.relu(self.fc1_rh3)
        if train_mode == True:
            self.relu1_rh3 = tf.nn.dropout(self.relu1_rh3, 0.7)

        self.concat_rh = tf.concat([self.relu1_rh1, self.relu1_rh2, self.relu1_rh3], axis=-1)
        self.fc2_rh = self.fc_layer(self.concat_rh, 512 * 3, 1024, "fc_2_rh")
        self.relu2_rh = tf.nn.relu(self.fc2_rh)
        if train_mode == True:
            self.relu2_rh = tf.nn.dropout(self.relu2_rh, 0.5)



        self.fc1_mh1 = self.fc_layer(self.pool_m1, np.prod([int(x) for x in self.pool_m1.get_shape()[1:]]), 512,
                                     "fc_1_mh_1")
        self.relu1_mh1 = tf.nn.relu(self.fc1_mh1)
        if train_mode == True:
            self.relu1_mh1 = tf.nn.dropout(self.relu1_mh1, 0.7)

        self.fc1_mh2 = self.fc_layer(self.pool_m2, np.prod([int(x) for x in self.pool_m2.get_shape()[1:]]), 512,
                                     "fc_1_mh_2")
        self.relu1_mh2 = tf.nn.relu(self.fc1_mh2)
        if train_mode == True:
            self.relu1_mh2 = tf.nn.dropout(self.relu1_mh2, 0.7)

        self.fc1_mh3 = self.fc_layer(self.pool_m3, np.prod([int(x) for x in self.pool_m3.get_shape()[1:]]), 512,
                                     "fc_1_mh_3")
        self.relu1_mh3 = tf.nn.relu(self.fc1_mh3)
        if train_mode == True:
            self.relu1_mh3 = tf.nn.dropout(self.relu1_mh3, 0.7)

        self.concat_mh = tf.concat([self.relu1_mh1, self.relu1_mh2, self.relu1_mh3], axis=-1)
        self.fc2_mh = self.fc_layer(self.concat_mh, 512 * 3, 1024, "fc_2_mh")
        self.relu2_mh = tf.nn.relu(self.fc2_mh)
        if train_mode == True:
            self.relu2_mh = tf.nn.dropout(self.relu2_mh, 0.5)



        self.fc1_ih1 = self.fc_layer(self.pool_i1, np.prod([int(x) for x in self.pool_i1.get_shape()[1:]]), 512,
                                     "fc_1_ih_1")
        self.relu1_ih1 = tf.nn.relu(self.fc1_ih1)
        if train_mode == True:
            self.relu1_ih1 = tf.nn.dropout(self.relu1_ih1, 0.7)

        self.fc1_ih2 = self.fc_layer(self.pool_i2, np.prod([int(x) for x in self.pool_i2.get_shape()[1:]]), 512,
                                     "fc_1_ih_2")
        self.relu1_ih2 = tf.nn.relu(self.fc1_ih2)
        if train_mode == True:
            self.relu1_ih2 = tf.nn.dropout(self.relu1_ih2, 0.7)

        self.fc1_ih3 = self.fc_layer(self.pool_i3, np.prod([int(x) for x in self.pool_i3.get_shape()[1:]]), 512,
                                     "fc_1_ih_3")
        self.relu1_ih3 = tf.nn.relu(self.fc1_ih3)
        if train_mode == True:
            self.relu1_ih3 = tf.nn.dropout(self.relu1_ih3, 0.7)

        self.concat_ih = tf.concat([self.relu1_ih1, self.relu1_ih2, self.relu1_ih3], axis=-1)
        self.fc2_ih = self.fc_layer(self.concat_ih, 512 * 3, 1024, "fc_2_ih")
        self.relu2_ih = tf.nn.relu(self.fc2_ih)
        if train_mode == True:
            self.relu2_ih = tf.nn.dropout(self.relu2_ih, 0.5)



        self.fc1_th1 = self.fc_layer(self.pool_t1, np.prod([int(x) for x in self.pool_t1.get_shape()[1:]]), 512,
                                     "fc_1_th_1")
        self.relu1_th1 = tf.nn.relu(self.fc1_th1)
        if train_mode == True:
            self.relu1_th1 = tf.nn.dropout(self.relu1_th1, 0.7)

        self.fc1_th2 = self.fc_layer(self.pool_t2, np.prod([int(x) for x in self.pool_t2.get_shape()[1:]]), 512,
                                     "fc_1_th_2")
        self.relu1_th2 = tf.nn.relu(self.fc1_th2)
        if train_mode == True:
            self.relu1_th2 = tf.nn.dropout(self.relu1_th2, 0.7)

        self.fc1_th3 = self.fc_layer(self.pool_t3, np.prod([int(x) for x in self.pool_t3.get_shape()[1:]]), 512,
                                     "fc_1_th_3")
        self.relu1_th3 = tf.nn.relu(self.fc1_th3)
        if train_mode == True:
            self.relu1_th3 = tf.nn.dropout(self.relu1_th3, 0.7)

        self.concat_th = tf.concat([self.relu1_th1, self.relu1_th2, self.relu1_th3], axis=-1)
        self.fc2_th = self.fc_layer(self.concat_th, 512 * 3, 1024, "fc_2_th")
        self.relu2_th = tf.nn.relu(self.fc2_th)
        if train_mode == True:
            self.relu2_th = tf.nn.dropout(self.relu2_th, 0.5)



        self.h_concat = tf.concat([self.relu2_ph, self.relu2_rh, self.relu2_mh, self.relu2_ih, self.relu2_th], axis=-1)
        self.final_fc1 = self.fc_layer(self.h_concat, 1024 * 5, 1024, "final_fc_1")
        self.final_relu1 = tf.nn.relu(self.final_fc1)
        if train_mode == True:
            self.final_fc1 = tf.nn.dropout(self.final_relu1, 0.5)
        self.final_fc2 = self.fc_layer(self.final_relu1, 1024, output_shape, "final_fc_2")
        self.output = tf.identity(self.final_fc2)





























    def batchnorm(self, layer):
        m, v = tf.nn.moments(layer, [0])
        return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool_4(self,bottom,name):
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