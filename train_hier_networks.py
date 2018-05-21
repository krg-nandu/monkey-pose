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
            model=hier_model_struct()
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
            val_model=hier_model_struct()
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
            model=hier_model_struct()
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
                        result_name="results_com/hier/results/image_{}.png".format(step)
                        save_result_image(images_np,images_coms,images_Ms,labels_sp,results_sp,seqconfig['cube'][2] / 2.,result_name)
                    if joint_error >40:
                        result_name = "results_com/hier/bad/image_{}.png".format(step)
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

            pickleCache = 'results_com/hier/cnn_result_cache.pkl'
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((joint_labels, joint_results), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            np_labels = np.asarray(joint_labels)
            np_results = np.asarray(joint_results)
            np_mean = getMeanError_np(np_labels, np_results)
            print np_mean


class hier_model_struct:
    def __init__(self,trainable=True):
        self.trainable=trainable
        self.data_dict = None
        self.var_dict = {}
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(self,depth,output_shape,P_shape,R_shape,M_shape,I_shape,T_shape,batch_norm=None,train_mode=None):
        print("hierarchical cnn network")
        input_image=tf.identity(depth,name="lr_input")
        self.conv1=self.conv_layer(input_image,int(input_image.get_shape().as_list()[-1]),64,"conv_1",filter_size=3,batchnorm=batch_norm)
        self.pool1 = self.max_pool(self.conv1, 'pool_1')

        self.conv2 = self.conv_layer(self.pool1, 64, 128, "conv_2", filter_size=3,batchnorm=batch_norm)
        self.pool2 = self.max_pool(self.conv2, 'pool_2')

        # finger P, R
        self.pr_conv3 =self.conv_layer(self.pool2,128,256,"pr_con_3",filter_size=3, batchnorm=batch_norm)
        self.pr_pool3 =self.max_pool(self.pr_conv3,'pr_pool_3')

        self.pr_conv4 = self.conv_layer(self.pr_pool3, 256, 512, "pr_con_4", filter_size=3, batchnorm=batch_norm)
        self.pr_pool4 = self.max_pool(self.pr_conv4, 'pr_pool_4')

        ## finger P
        self.p_conv5 =self.conv_layer(self.pr_pool4,512,512,"p_con_5",filter_size=3,batchnorm=batch_norm)
        self.p_pool5=self.max_pool(self.p_conv5,'p_pool_5')

        self.p_conv6 = self.conv_layer(self.p_pool5,512,1024,'p_con_6',filter_size=5,batchnorm=batch_norm)
        self.p_pool6=self.max_pool(self.p_conv6,'p_pool_6')

        self.p_fc1=self.fc_layer(self.p_pool6,np.prod([int(x) for x in self.p_pool6.get_shape().as_list()[1:]]),1024,"p_fc_1")
        self.p_relu1=tf.nn.relu(self.p_fc1)
        if train_mode==True:
            self.p_relu1=tf.nn.dropout(self.p_relu1,0.7)

        self.p_fc2=self.fc_layer(self.p_relu1,1024,1024,"p_fc_2")
        self.p_relu2=tf.nn.relu(self.p_fc2)
        if train_mode==True:
            self.p_relu2=tf.nn.dropout(self.p_relu2,0.5)

        self.p_fc3=self.fc_layer(self.p_relu2,1024,P_shape,"p_fc_3")
        self.p_output=tf.identity(self.p_fc3,name="p_out_put")

        ## finger R
        self.r_conv5 = self.conv_layer(self.pr_pool4, 512, 512, "r_con_5", filter_size=3, batchnorm=batch_norm)
        self.r_pool5 = self.max_pool(self.r_conv5, 'r_pool_5')

        self.r_conv6 = self.conv_layer(self.r_pool5, 512,1024,'r_con_6', filter_size=5, batchnorm=batch_norm)
        self.r_pool6 = self.max_pool(self.r_conv6, 'r_pool_6')

        self.r_fc1 = self.fc_layer(self.r_pool6, np.prod([int(x) for x in self.r_pool6.get_shape().as_list()[1:]]), 1024,
                                   "r_fc_1")
        self.r_relu1 = tf.nn.relu(self.r_fc1)
        if train_mode == True:
            self.r_relu1 = tf.nn.dropout(self.r_relu1, 0.7)

        self.r_fc2 = self.fc_layer(self.r_relu1, 1024, 1024, "r_fc_2")
        self.r_relu2 = tf.nn.relu(self.r_fc2)
        if train_mode == True:
            self.r_relu2 = tf.nn.dropout(self.r_relu2, 0.5)

        self.r_fc3 = self.fc_layer(self.r_relu2, 1024, R_shape, "r_fc_3")
        self.r_output = tf.identity(self.r_fc3, name="r_out_put")

        # finger M, I
        self.mi_conv3 = self.conv_layer(self.pool2, 128, 256, "mi_con_3", filter_size=3, batchnorm=batch_norm)
        self.mi_pool3 = self.max_pool(self.mi_conv3, 'mi_pool_3')

        self.mi_conv4 = self.conv_layer(self.mi_pool3, 256, 512, "mi_con_4", filter_size=3, batchnorm=batch_norm)
        self.mi_pool4 = self.max_pool(self.mi_conv4, 'mi_pool_4')

        ## finger M
        self.m_conv5 = self.conv_layer(self.mi_pool4, 512, 512, "m_con_5", filter_size=3, batchnorm=batch_norm)
        self.m_pool5 = self.max_pool(self.m_conv5, 'm_pool_5')

        self.m_conv6 = self.conv_layer(self.m_pool5, 512, 1024, 'm_con_6', filter_size=5, batchnorm=batch_norm)
        self.m_pool6 = self.max_pool(self.m_conv6, 'm_pool_6')

        self.m_fc1 = self.fc_layer(self.m_pool6, np.prod([int(x) for x in self.m_pool6.get_shape().as_list()[1:]]), 1024,
                                   "m_fc_1")
        self.m_relu1 = tf.nn.relu(self.m_fc1)
        if train_mode == True:
            self.m_relu1 = tf.nn.dropout(self.m_relu1, 0.7)

        self.m_fc2 = self.fc_layer(self.m_relu1, 1024, 1024, "m_fc_2")
        self.m_relu2 = tf.nn.relu(self.m_fc2)
        if train_mode == True:
            self.m_relu2 = tf.nn.dropout(self.m_relu2, 0.5)

        self.m_fc3 = self.fc_layer(self.m_relu2, 1024, M_shape, "m_fc_3")
        self.m_output = tf.identity(self.m_fc3, name="m_out_put")

        ## finger I
        self.i_conv5 = self.conv_layer(self.mi_pool4, 512, 512, "i_con_5", filter_size=3, batchnorm=batch_norm)
        self.i_pool5 = self.max_pool(self.i_conv5, 'i_pool_5')

        self.i_conv6 = self.conv_layer(self.i_pool5, 512, 1024, 'i_con_6', filter_size=5, batchnorm=batch_norm)
        self.i_pool6 = self.max_pool(self.i_conv6, 'i_pool_6')

        self.i_fc1 = self.fc_layer(self.i_pool6, np.prod([int(x) for x in self.i_pool6.get_shape().as_list()[1:]]), 1024,
                                   "i_fc_1")
        self.i_relu1 = tf.nn.relu(self.i_fc1)
        if train_mode == True:
            self.i_relu1 = tf.nn.dropout(self.i_relu1, 0.7)

        self.i_fc2 = self.fc_layer(self.i_relu1, 1024, 1024, "i_fc_2")
        self.i_relu2 = tf.nn.relu(self.i_fc2)
        if train_mode == True:
            self.i_relu2 = tf.nn.dropout(self.i_relu2, 0.5)

        self.i_fc3 = self.fc_layer(self.i_relu2, 1024, I_shape, "i_fc_3")
        self.i_output = tf.identity(self.i_fc3, name="i_out_put")

        # finger T
        self.t_conv3 = self.conv_layer(self.pool2, 128, 256, "t_con_3", filter_size=3, batchnorm=batch_norm)
        self.t_pool3 = self.max_pool(self.t_conv3, 't_pool_3')

        self.t_conv4 = self.conv_layer(self.t_pool3, 256, 512, "t_con_4", filter_size=3, batchnorm=batch_norm)
        self.t_pool4 = self.max_pool(self.t_conv4, 't_pool_4')

        self.t_conv5 = self.conv_layer(self.t_pool4, 512, 512, "t_con_5", filter_size=3, batchnorm=batch_norm)
        self.t_pool5 = self.max_pool(self.t_conv5, 't_pool_5')

        self.t_conv6 = self.conv_layer(self.t_pool5, 512, 1024, 't_con_6', filter_size=5, batchnorm=batch_norm)
        self.t_pool6 = self.max_pool(self.t_conv6, 't_pool_6')

        self.t_fc1 = self.fc_layer(self.t_pool6, np.prod([int(x) for x in self.t_pool6.get_shape().as_list()[1:]]), 1024,
                                   "t_fc_1")
        self.t_relu1 = tf.nn.relu(self.t_fc1)
        if train_mode == True:
            self.t_relu1 = tf.nn.dropout(self.t_relu1, 0.7)

        self.t_fc2 = self.fc_layer(self.t_relu1, 1024, 1024, "t_fc_2")
        self.t_relu2 = tf.nn.relu(self.t_fc2)
        if train_mode == True:
            self.t_relu2 = tf.nn.dropout(self.t_relu2, 0.5)

        self.t_fc3 = self.fc_layer(self.t_relu2, 1024, T_shape, "t_fc_3")
        self.t_output = tf.identity(self.t_fc3, name="t_out_put")

        # whole hand

        self.ph_fc1=self.fc_layer(self.p_pool6,np.prod([int(x) for x in self.p_pool6.get_shape().as_list()[1:]]),1024,"ph_fc_1")
        self.ph_relu1=tf.nn.relu(self.ph_fc1)
        if train_mode==True:
            self.ph_relu1=tf.nn.dropout(self.ph_relu1,0.7)
        self.ph_fc2=self.fc_layer(self.ph_relu1,1024,1024,"ph_fc_2")
        self.ph_relu2=tf.nn.relu(self.ph_fc2)
        if train_mode==True:
            self.ph_relu2=tf.nn.dropout(self.ph_relu2,0.5)


        self.rh_fc1 = self.fc_layer(self.r_pool6, np.prod([int(x) for x in self.r_pool6.get_shape().as_list()[1:]]), 1024,
                                    "rh_fc_1")
        self.rh_relu1 = tf.nn.relu(self.rh_fc1)
        if train_mode == True:
            self.rh_relu1 = tf.nn.dropout(self.rh_relu1, 0.7)
        self.rh_fc2 = self.fc_layer(self.rh_relu1, 1024, 1024, "rh_fc_2")
        self.rh_relu2 = tf.nn.relu(self.rh_fc2)
        if train_mode == True:
            self.rh_relu2 = tf.nn.dropout(self.rh_relu2, 0.5)


        self.mh_fc1 = self.fc_layer(self.m_pool6, np.prod([int(x) for x in self.m_pool6.get_shape().as_list()[1:]]), 1024,
                                    "mh_fc_1")
        self.mh_relu1 = tf.nn.relu(self.mh_fc1)
        if train_mode == True:
            self.mh_relu1 = tf.nn.dropout(self.mh_relu1, 0.7)
        self.mh_fc2 = self.fc_layer(self.mh_relu1, 1024, 1024, "mh_fc_2")
        self.mh_relu2 = tf.nn.relu(self.mh_fc2)
        if train_mode == True:
            self.mh_relu2 = tf.nn.dropout(self.mh_relu2, 0.5)

        self.ih_fc1 = self.fc_layer(self.i_pool6, np.prod([int(x) for x in self.i_pool6.get_shape().as_list()[1:]]), 1024,
                                    "ih_fc_1")
        self.ih_relu1 = tf.nn.relu(self.ih_fc1)
        if train_mode == True:
            self.ih_relu1 = tf.nn.dropout(self.ih_relu1, 0.7)
        self.ih_fc2 = self.fc_layer(self.ih_relu1, 1024, 1024, "ih_fc_2")
        self.ih_relu2 = tf.nn.relu(self.ih_fc2)
        if train_mode == True:
            self.ih_relu2 = tf.nn.dropout(self.ih_relu2, 0.5)

        self.th_fc1 = self.fc_layer(self.t_pool6, np.prod([int(x) for x in self.t_pool6.get_shape().as_list()[1:]]), 1024,
                                    "th_fc_1")
        self.th_relu1 = tf.nn.relu(self.th_fc1)
        if train_mode == True:
            self.th_relu1 = tf.nn.dropout(self.th_relu1, 0.7)
        self.th_fc2 = self.fc_layer(self.th_relu1, 1024, 1024, "th_fc_2")
        self.th_relu2 = tf.nn.relu(self.th_fc2)
        if train_mode == True:
            self.th_relu2 = tf.nn.dropout(self.th_relu2, 0.5)

        self.h_concat=tf.concat([self.ph_relu2,self.rh_relu2,self.mh_relu2,self.ih_relu2,self.th_relu2],axis=-1)
        self.final_fc1=self.fc_layer(self.h_concat,1024*5,1024,"final_fc_1")
        self.final_relu1=tf.nn.relu(self.final_fc1)
        if train_mode ==True:
            self.final_fc1=tf.nn.dropout(self.final_relu1,0.5)
        self.final_fc2=self.fc_layer(self.final_relu1,1024,output_shape,"final_fc_2")
        self.output=tf.identity(self.final_fc2)





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