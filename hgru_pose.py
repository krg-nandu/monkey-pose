import numpy as np
import tensorflow as tf
import hgru_module


class model:

    def __init__(self,trainable=True):
        self.trainable=trainable
        self.data_dict = None
        self.var_dict = {}
        self.SRF = 1
        self.SSN = 15
        self.SSF = 15
        self.strides = [1, 1, 1, 1]
        self._BATCH_NORM_DECAY = 0.997
        self._BATCH_NORM_EPSILON = 1e-5
        self.padding = 'SAME'
        self.timesteps = 8
        self.aux = {
            'recurrent_nl': 'tanh',
            'rectify_weights': None,  # False,
            'pre_batchnorm': False,
            'gate_filter': 1,
            'xi': False,
            'post_batchnorm': False,
            'dense_connections': False,
            'symmetric_weights': True,  # Lateral weight sharing
            'symmetric_gate_weights': False,
            'batch_norm': False,
            'atrous_convolutions': False,
            'output_gru_gates': False,
            'association_field': True,
            'multiplicative_excitation': True,
            'gru_gates': True,
            'gamma': True,
            'adapation': True,
            'trainable': True,
        }

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(self,depth,output_shape,batch_norm=None,train_mode=None):
        print ("cnn network")
        input_image=tf.identity(depth, name="lr_input")
        self.conv1=self.conv_layer(input_image,int(input_image.get_shape()[-1]),64,"conv_1",filter_size=3)
        self.pool1=self.max_pool(self.conv1, 'pool_1')
        self.pool1 = tf.layers.batch_normalization(
            inputs=self.pool1,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)
        self.conv2=self.conv_layer(self.pool1,64,64,"conv_2",filter_size=3)
        self.conv2 = tf.layers.batch_normalization(
            inputs=self.conv2,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)
        self.conv3=self.conv_layer(self.conv2,64,64,"conv_3",filter_size=3)
        self.conv3 = tf.layers.batch_normalization(
            inputs=self.conv3,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)
        self.hgru=self.hgru_layer(self.conv3)
        self.hgru = tf.layers.batch_normalization(
            inputs=self.hgru,
            axis=3,
            momentum=self._BATCH_NORM_DECAY,
            epsilon=self._BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=train_mode,
            fused=True)
        self.fc1=self.fc_layer(self.hgru,np.prod([int(x) for x in self.hgru.get_shape()[1:]]),1024,"fc_1")
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
        self.fc4=self.fc_layer(self.relu3,1024,output_shape,"fc_out")
        self.out_put=tf.identity(self.fc4,name="out_put")

    def hgru_layer(self, bottom):
        """Contextual model from paper with frozen U & eCRFs."""
        contextual_layer = hgru_module.ContextualCircuit(
            X=bottom,
            timesteps=self.timesteps,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding,
            aux=self.aux)
        return contextual_layer.build()

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