import os
import tensorflow as tf
from data_loader import inputs
from check_fun import showdepth, showImagefromArray,showImageLable,trans3DsToImg,showImageLableCom,showImageJoints,showImageJointsandResults, showJointsOnly
import glob
from pose_evaluation import getMeanError,getMeanError_np,getMean_np,getMeanError_train
import numpy as np
import cPickle
import tf_monkeydetector
import matplotlib.pyplot as plt
import hgru_pose


def check_image_label(im, jts, com, M,cube_22,allJoints=False,line=False):
    relen=len(jts)/3
    jt = jts.reshape((relen, 3))
    jtorig = jt * cube_22
    jcrop = trans3DsToImg(jtorig, com, M)
    showImageJoints(im,jcrop,allJoints=allJoints,line=line)

def save_result_image(images_np,images_coms,images_Ms,labels_np,images_results,cube_22,name,line=True):
    val_im = images_np[0].reshape([128, 128])
    if labels_np.ndim > 2:
        jtorig=labels_np[0]
        re_jtorig = images_results[0]
    else:
        jtorig = labels_np
        re_jtorig = images_results
    com = images_coms[0]
    M=images_Ms[0]
    jcrop = trans3DsToImg(jtorig,com,M)
    re_jcrop = trans3DsToImg(re_jtorig, com, M)

    showImageJointsandResults(val_im,jcrop,re_jcrop,save=True,imagename=name,line=False,allJoints=True)

def calc_com_error(labels,predictions):
    # note that here there are only 2 dimensions -- batch index and (u,v,d)
    assert (labels.shape == predictions.shape)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels-predictions), 1)),0)

def prepare_data(image_np,image_label_shaped,tr_res,md,config, show=False):
    patches = np.zeros(
        (config.train_batch, config.image_target_size[0], config.image_target_size[1], config.image_target_size[2]))
    rel_labels = np.zeros((config.train_batch, config.num_joints * config.num_dims))
    for im in range(image_np.shape[0]):
        # get the cropped patches and normalize the image/labels
        dpt, M, com = md.cropArea3D(image_np[im] * config.image_max_depth,
                                    com=tr_res[im] * [config.image_orig_size[0], config.image_orig_size[1],
                                                      config.image_max_depth])
        patches[im] = np.expand_dims(dpt, axis=2) / config.image_max_depth
        # get the relative coordinates
        jnts_uvd = md.xyztouvd_np(image_label_shaped[im])
        rel_jnts_xyz, rel_jnts_uvd = md.getRelativeCoordinates(image_label_shaped[im], jnts_uvd, com, M)
        rel_labels[im] = np.clip(
            np.asarray(np.reshape(rel_jnts_xyz, (config.num_joints * config.num_dims,)), dtype='float32') / (
            md.cube[2] / 2.), -1, 1)
        if show:
            plt.clf(); plt.imshow(dpt); plt.scatter(rel_jnts_uvd[:,0],rel_jnts_uvd[:,1]); plt.pause(0.001)
    return patches, rel_labels

def prepare_data_test(image_np,tr_res,md,config):
    patches = np.zeros(
        (config.test_batch, config.image_target_size[0], config.image_target_size[1], config.image_target_size[2]))
    coms = []
    Ms = []
    for im in range(image_np.shape[0]):
        # get the cropped patches and normalize the image/labels
        dpt, M, com = md.cropArea3D(image_np[im] * config.image_max_depth,
                                    com=tr_res[im] * [config.image_orig_size[0], config.image_orig_size[1],
                                                      config.image_max_depth])
        patches[im] = np.expand_dims(dpt, axis=2) / config.image_max_depth
        coms.append(com)
        Ms.append(M)
    return patches, coms, Ms

def train_model(config,seqconfig):
    md = tf_monkeydetector.tfMonkeyDetector(365.456,365.456,256,212,[800,800,1200],200,10000)
    phaseII = False

    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    val_data = os.path.join(config.tfrecord_dir, config.val_tfrecords)
    with tf.device('/cpu:0'):
        train_images, train_labels = inputs(tfrecord_file = train_data,
                                           num_epochs=config.epochs,
                                           image_target_size = config.image_orig_size,
                                           label_shape=config.num_classes,
                                           batch_size =config.train_batch,
                                           data_augment=False)
        val_images, val_labels = inputs(tfrecord_file=val_data,
                                                            num_epochs=config.epochs,
                                                            image_target_size=config.image_orig_size,
                                                            label_shape=config.num_classes,
                                                            batch_size=config.val_batch)

    with tf.device('/gpu:0'):
        with tf.variable_scope("cnn") as scope:
            # place holders for training and validation!
            crop_input = tf.placeholder(tf.float32,
                                        [None, config.image_target_size[0], config.image_target_size[1],
                                         config.image_target_size[2]], name='patch_placeholder')
            crop_labels = tf.placeholder(tf.float32,
                                        [None, config.num_joints*config.num_dims], name='labels_placeholder')

            val_crop_input = tf.placeholder(tf.float32,
                                        [None, config.image_target_size[0], config.image_target_size[1],
                                         config.image_target_size[2]], name='patch_placeholder')
            val_crop_labels = tf.placeholder(tf.float32,
                                         [None, config.num_joints * config.num_dims], name='labels_placeholder')

            print("create training graphs:")
            ##########
            # PHASE I: attention component to locate the center of mass of the monkey
            ##########
            # build the model
            attn_model = attn_model_struct()
            train_images = train_images / config.image_max_depth
            attn_model.build(train_images,config.num_dims,train_mode=True)
            # calculate the 3d center of mass and project to image coordinates and normalize!
            train_labels_reshaped = tf.reshape(train_labels, [config.train_batch, config.num_classes / 3, 3])
            com2d = md.calculateCoMfrom3DJoints(train_labels_reshaped) / [config.image_orig_size[0],config.image_orig_size[1],config.image_max_depth]
            # define the loss and optimization functions
            attn_loss = tf.nn.l2_loss(attn_model.out_put - com2d)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                attn_train_op = tf.train.AdamOptimizer(1e-4).minimize(attn_loss)
            # if config.wd_penalty is None:
            #     attn_train_op = tf.train.AdamOptimizer(1e-4).minimize(attn_loss)
            # else:
            #     attn_wd_l = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'biases' not in v.name]
            #     attn_loss_wd=attn_loss+(config.wd_penalty * tf.add_n([tf.nn.l2_loss(x) for x in attn_wd_l]))
            #     attn_train_op = tf.train.AdamOptimizer(1e-4).minimize(attn_loss_wd)

            attn_train_results_shaped = tf.reshape(attn_model.out_put, [config.val_batch, config.num_dims])
            attn_train_error = calc_com_error(com2d,attn_train_results_shaped)

            ##########
            # PHASE II: intrinsic pose estimation component
            ##########
            # build the second model
            # model=cnn_model_struct()
            model=hgru_pose.model()
            model.build(crop_input,config.num_classes,train_mode=True)
            loss=tf.nn.l2_loss(model.out_put-crop_labels)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
            # if config.wd_penalty is None:
            #     train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
            # else:
            #     wd_l = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'biases' not in v.name]
            #     loss_wd=loss+(config.wd_penalty * tf.add_n([tf.nn.l2_loss(x) for x in wd_l]))
            #     train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_wd)

            crop_labels_shaped=tf.reshape(crop_labels,[config.train_batch,config.num_classes/3,3])*seqconfig['cube'][2] / 2.
            train_results_shaped=tf.reshape(model.out_put,[config.train_batch,config.num_classes/3,3])*seqconfig['cube'][2] / 2.
            train_error = getMeanError_train(crop_labels_shaped,train_results_shaped)

            print ('using validation!')
            scope.reuse_variables()
            attn_val_model = attn_model_struct()
            val_images = val_images / config.image_max_depth
            attn_val_model.build(val_images,config.num_dims,train_mode=False)
            attn_val_labels_shaped = tf.reshape(val_labels,[config.val_batch,config.num_classes/3,3])
            #attn_val_results_shaped = tf.reshape(attn_val_model.out_put,[config.val_batch,config.num_dims])*[config.image_orig_size[0],config.image_orig_size[1],config.image_max_depth]
            attn_val_results_shaped = tf.reshape(attn_val_model.out_put, [config.val_batch, config.num_dims])
            val_com2d = md.calculateCoMfrom3DJoints(attn_val_labels_shaped) / [config.image_orig_size[0],config.image_orig_size[1],config.image_max_depth]
            attn_val_error = calc_com_error(val_com2d, attn_val_results_shaped)

            val_model=cnn_model_struct()
            val_model.build(val_crop_input,config.num_classes,train_mode=False)
            val_crop_labels_shaped = tf.reshape(val_crop_labels, [config.val_batch, config.num_classes / 3, 3])*seqconfig['cube'][2] / 2.
            val_results_shaped = tf.reshape(val_model.out_put, [config.val_batch, config.num_classes / 3, 3])*seqconfig['cube'][2] / 2.
            val_error = getMeanError_train(val_crop_labels_shaped, val_results_shaped)

            tf.summary.scalar("attention_loss", attn_loss)
            tf.summary.scalar("train error", attn_train_error)
            tf.summary.scalar("validation error", attn_val_error)

            if phaseII:
                tf.summary.scalar("pose_loss", loss)
                #if config.wd_penalty is not None:
                #    tf.summary.scalar("pose_loss_wd", loss_wd)
                tf.summary.scalar("pose_train error", train_error)
                tf.summary.scalar("pose_validation error", val_error)

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
                if step < config.num_attn_steps:
                    _, image_np, image_label, tr_error, tr_loss, tr_res, tr_com = sess.run(
                        [attn_train_op, train_images, train_labels, attn_train_error, attn_loss, attn_train_results_shaped, com2d])
                    print("step={},loss={},error={}".format(step, tr_loss, tr_error))

                    if step % 250 ==0:
                        val_image_np, val_image_label, v_error, v_com, v_res= sess.run([val_images, val_labels, attn_val_error, val_com2d, attn_val_results_shaped])
                        print("     val error={}".format(v_error))
                        summary_str=sess.run(summary_op)
                        summary_writer.add_summary(summary_str,step)
                else:
                    phaseII = True
                    # get results from phase one for the center of mass
                    image_np, image_label_shaped, tr_error, tr_loss, tr_res, tr_com = sess.run(
                        [train_images, train_labels_reshaped, attn_train_error, attn_loss,
                         attn_train_results_shaped, com2d])

                    patches, rel_labels = prepare_data(image_np,image_label_shaped,tr_res,md,config)
                    #import ipdb; ipdb.set_trace()
                    # _, tr_error, tr_loss, tr_loss_wd = sess.run(
                    #     [train_op, train_error, loss, loss_wd],feed_dict={crop_input:patches,crop_labels:rel_labels})
                    _, tr_error, tr_loss = sess.run(
                         [train_op, train_error, loss],feed_dict={crop_input:patches,crop_labels:rel_labels})
                    print("step={},loss={},error={} mm".format(step, tr_loss, tr_error))

                    if step % 1000 == 0:
                        # run validation
                        val_image_np, val_image_label_shaped, _, val_res, _ = sess.run(
                            [val_images, attn_val_labels_shaped, attn_val_error,
                             attn_val_results_shaped, val_com2d])

                        patches, rel_labels = prepare_data(val_image_np, val_image_label_shaped, val_res, md, config, show=True)
                        v_error = sess.run([val_error],feed_dict={val_crop_input: patches, val_crop_labels: rel_labels})
                        print("step={},error={} mm".format(step, v_error))

                        # save the model checkpoint if it's the best yet
                        if first_v is True:
                            val_min = v_error
                            first_v = False
                        else:
                            if v_error < val_min:
                                print(os.path.join(
                                    config.model_output,
                                    'attn_cnn_model' + str(step) +'.ckpt'))
                                saver.save(sess, os.path.join(
                                    config.model_output,
                                    'attn_cnn_model' + str(step) +'.ckpt'), global_step=step)
                                # store the new max validation accuracy
                                val_min = v_error

                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                step += 1
        except tf.errors.OutOfRangeError:
            print("Done. Epoch limit reached.")
        finally:
            coord.request_stop()
        coord.join(threads)

def test_model(config,seqconfig):
    #test_data = os.path.join(config.tfrecord_dir, config.test_tfrecords)
    test_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    
    md = tf_monkeydetector.tfMonkeyDetector(365.456, 365.456, 256, 212, [800, 800, 1200], 200, 10000)
    print test_data
    with tf.device('/cpu:0'):
        images, labels = inputs(tfrecord_file=test_data,
                                            num_epochs=None,
                                            image_target_size=config.image_orig_size,
                                            label_shape=config.num_classes,
                                            batch_size=config.test_batch)

    with tf.device('/gpu:0'):
        with tf.variable_scope("cnn") as scope:

            crop_input = tf.placeholder(tf.float32,
                                        [None, config.image_target_size[0], config.image_target_size[1],
                                         config.image_target_size[2]], name='patch_placeholder')
            crop_labels = tf.placeholder(tf.float32,
                                         [None, config.num_joints * config.num_dims], name='labels_placeholder')

            attn_model = attn_model_struct()
            images = images / config.image_max_depth
            attn_model.build(images, config.num_dims, train_mode=True)
            attn_results_shaped = tf.reshape(attn_model.out_put, [1, config.num_dims])

            model=cnn_model_struct()
            model.build(crop_input, config.num_classes, train_mode=False)
            labels_shaped = tf.reshape(labels, [(config.num_classes / 3), 3]) * \
                                seqconfig['cube'][2] / 2.
            results_shaped = tf.reshape(model.out_put, [(config.num_classes / 3), 3]) * \
                                 seqconfig['cube'][2] / 2.
            #error = getMeanError(labels_shaped, results_shaped)

        # Initialize the graph
        gpuconfig = tf.ConfigProto()
        gpuconfig.gpu_options.allow_growth = True
        gpuconfig.allow_soft_placement = True
        saver = tf.train.Saver()

        with tf.Session(config=gpuconfig).as_default() as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            step=0
            coord = tf.train.Coordinator()
            threads=tf.train.start_queue_runners(coord=coord,sess=sess)

            checkpoints = tf.train.latest_checkpoint(config.model_output)
            saver.restore(sess, checkpoints)

            try:
                while not coord.should_stop():
                    #import ipdb; ipdb.set_trace()
                    images_np,results_com = sess.run([images,attn_results_shaped])
                    patches,coms,Ms = prepare_data_test(images_np, results_com, md, config)

                    t_res = sess.run([results_shaped], feed_dict={crop_input: patches})
                    #print("step={}, test error={} mm".format(step,joint_error))
                    print("step={}".format(step))

                    #if step%100 ==0:
                    result_name="{}image_{}.png".format(config.results_dir,step)
                    retrieved_jnts_xyz, retrieved_jnts_uvd = md.getAbsoluteCoordinates(t_res[0], coms[0])
                    plt.imshow(images_np[0].squeeze())
                    plt.scatter(retrieved_jnts_uvd[:, 0], retrieved_jnts_uvd[:, 1], c='r')
                    plt.savefig(result_name)
                    plt.show()

                    step+=1
            except tf.errors.OutOfRangeError:
                print("Done.Epoch limit reached.")
            finally:
                coord.request_stop()
            coord.join(threads)
            print("load model from {}".format(checkpoints))


def eval_model_on_real_data(config, seqconfig):
    # test_data = os.path.join(config.tfrecord_dir, config.test_tfrecords)
    test_data = glob.glob(config.real_data_dir+'*.npy')
    md = tf_monkeydetector.tfMonkeyDetector(365.456, 365.456, 256, 212, [800, 800, 1200], 200, 10000)

    with tf.device('/gpu:0'):
        with tf.variable_scope("cnn") as scope:

            input_image = tf.placeholder(tf.float32,
                                        [None, config.image_orig_size[0], config.image_orig_size[1],
                                         config.image_orig_size[2]], name='input_placeholder')

            crop_input = tf.placeholder(tf.float32,
                                        [None, config.image_target_size[0], config.image_target_size[1],
                                         config.image_target_size[2]], name='patch_placeholder')

            attn_model = attn_model_struct()
            images = input_image / config.image_max_depth
            attn_model.build(images, config.num_dims, train_mode=True)
            attn_results_shaped = tf.reshape(attn_model.out_put, [1, config.num_dims])

            model = cnn_model_struct()
            model.build(crop_input, config.num_classes, train_mode=False)
            results_shaped = tf.reshape(model.out_put, [(config.num_classes / 3), 3]) * \
                             seqconfig['cube'][2] / 2.
            # error = getMeanError(labels_shaped, results_shaped)

        # Initialize the graph
        gpuconfig = tf.ConfigProto()
        gpuconfig.gpu_options.allow_growth = True
        gpuconfig.allow_soft_placement = True
        saver = tf.train.Saver()

        with tf.Session(config=gpuconfig).as_default() as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            step = 0
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            checkpoints = tf.train.latest_checkpoint(config.model_output)
            saver.restore(sess, checkpoints)

            try:
                while not coord.should_stop():
                    #import ipdb; ipdb.set_trace()
                    im = np.transpose(np.load(test_data[step]))
                    # preprocessing
                    im[(im < 1000)] = 10000
                    im[(im > 3000)] = 10000
                    im = np.asarray([im])

                    results_com = sess.run([attn_results_shaped],feed_dict={input_image:im})
                    results_com = results_com[0]
                    #results_com = np.asarray([[0.54245,0.2226,0.2551]])
                    #results_com = np.asarray([[0.9339, 0.2109, 0.1600]])

                    patches, coms, Ms = prepare_data_test(im/config.image_max_depth, results_com, md, config)

                    t_res = sess.run([results_shaped], feed_dict={crop_input: patches})
                    # print("step={}, test error={} mm".format(step,joint_error))
                    print("step={}".format(step))

                    # if step%100 ==0:
                    result_name = "{}image_{}.png".format(config.results_dir, step)
                    retrieved_jnts_xyz, retrieved_jnts_uvd = md.getAbsoluteCoordinates(t_res[0], coms[0])
                    plt.imshow(im[0].squeeze())
                    plt.scatter(retrieved_jnts_uvd[:, 0], retrieved_jnts_uvd[:, 1], c='r')
                    #plt.savefig(result_name)
                    plt.show()

                    step += 1
            except tf.errors.OutOfRangeError:
                print("Done.Epoch limit reached.")
            finally:
                coord.request_stop()
            coord.join(threads)
            print("load model from {}".format(checkpoints))


class attn_model_struct:
    def __init__(self,trainable=True):
        self.trainable=trainable
        self.data_dict = None
        self.var_dict = {}
        self._BATCH_NORM_DECAY = 0.997
        self._BATCH_NORM_EPSILON = 1e-5

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(self,depth,output_shape,batch_norm=None,train_mode=None):
        print ("attention network")
        input_image=tf.identity(depth, name="lr_input")
        input_image = tf.image.resize_images(input_image,[128,128])
        self.conv1=self.conv_layer(input_image,int(input_image.get_shape()[-1]),64,"aconv_1",filter_size=3)
        self.pool1=self.max_pool(self.conv1, 'apool_1')
        self.pool1 = tf.layers.batch_normalization(
            inputs=self.pool1,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)

        self.conv2=self.conv_layer(self.pool1,64,128,"aconv_2",filter_size=3)
        self.pool2=self.max_pool(self.conv2,'apool_2')
        self.pool2 = tf.layers.batch_normalization(
            inputs=self.pool2,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)

        self.conv3=self.conv_layer(self.pool2,128,256,"aconv_3",filter_size=3)
        self.pool3=self.max_pool(self.conv3,'pool_3')
        self.pool3 = tf.layers.batch_normalization(
            inputs=self.pool3,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)

        self.conv4=self.conv_layer(self.pool3,256,512,"aconv_4",filter_size=3)
        self.pool4=self.max_pool(self.conv4,'pool_4')
        self.pool4 = tf.layers.batch_normalization(
            inputs=self.pool4,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)

        self.conv5=self.conv_layer(self.pool4,512,1024,"aconv_5",filter_size=5)
        self.pool5=self.max_pool(self.conv5,'apool_5')
        self.pool5 = tf.layers.batch_normalization(
            inputs=self.pool5,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)

        self.fc1=self.fc_layer(self.pool5,np.prod([int(x) for x in self.pool5.get_shape()[1:]]),1024,"afc_1")
        self.relu1=tf.nn.relu(self.fc1)
        if train_mode==True:
            self.relu1=tf.nn.dropout(self.relu1,0.7)
        self.relu1 = tf.layers.batch_normalization(
            inputs=self.relu1,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)

        # self.fc2=self.fc_layer(self.relu1,1024,1024,"afc_2")
        # self.relu2=tf.nn.relu(self.fc2)
        # if train_mode==True:
        #     self.relu2=tf.nn.dropout(self.relu2,0.5)

        # self.fc3 = self.fc_layer(self.relu2, 1024, 1024, "afc_3")
        # self.relu3=tf.nn.relu(self.fc3)
        # if train_mode==True:
        #     self.relu3=tf.nn.dropout(self.relu3,0.5)

        self.fcout=self.fc_layer(self.relu1,1024,output_shape,"afc_out")
        self.out_put=tf.identity(self.fcout,name="a_output")

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


class cnn_model_struct:

    def __init__(self,trainable=True):
        self.trainable=trainable
        self.data_dict = None
        self.var_dict = {}

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(self,depth,output_shape,batch_norm=None,train_mode=None):
        print ("cnn network")
        input_image=tf.identity(depth, name="lr_input")
        self.conv1=self.conv_layer(input_image,int(input_image.get_shape()[-1]),64,"conv_1",filter_size=3)
        self.pool1=self.max_pool(self.conv1, 'pool_1')

        self.conv2=self.conv_layer(self.pool1,64,128,"conv_2",filter_size=3)
        self.pool2=self.max_pool(self.conv2,'pool_2')

        self.conv3=self.conv_layer(self.pool2,128,256,"conv_3",filter_size=3)
        self.pool3=self.max_pool(self.conv3,'pool_3')

        self.conv4=self.conv_layer(self.pool3,256,512,"conv_4",filter_size=3)
        self.pool4=self.max_pool(self.conv4,'pool_4')

        self.conv5=self.conv_layer(self.pool4,512,1024,"conv_5",filter_size=5)
        self.pool5=self.max_pool(self.conv5,'pool_5')

        self.fc1=self.fc_layer(self.pool5,np.prod([int(x) for x in self.pool5.get_shape()[1:]]),1024,"fc_1")
        self.relu1=tf.nn.relu(self.fc1)
        if train_mode==True:
            self.relu1=tf.nn.dropout(self.relu1,0.7)

        self.fc2=self.fc_layer(self.relu1,1024,1024,"fc_2")
        self.relu2=tf.nn.relu(self.fc2)
        if train_mode==True:
            self.relu2=tf.nn.dropout(self.relu2,0.5)

        self.fc3 = self.fc_layer(self.relu2, 1024, 1024, "fc_3")
        self.relu3=tf.nn.relu(self.fc3)
        if train_mode==True:
            self.relu3=tf.nn.dropout(self.relu3,0.5)

        self.fc4=self.fc_layer(self.relu3,1024,output_shape,"fc_4")
        self.out_put=tf.identity(self.fc4,name="out_put")


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