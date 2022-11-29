import tensorflow as tf

from deepsleep.nn import *

import pdb
class ConvEncNet(object):

    def __init__(
            self,
            batch_size,
            input_dims,
            n_classes,
            is_train,
            reuse_params,
            use_dropout,
            name="convEncNet"
    ):
        self.batch_size = batch_size
        self.featuremap_sizes = [128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256]
        self.filter_sizes= [7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 3, 3]
        self.strides = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.is_train = is_train
        self.reuse_params = reuse_params
        self.use_dropout = use_dropout
        self.name = name

        self.activations = []
        self.layer_idx = 1
        self.monitor_vars = []

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        self.sigma = tf.placeholder(
            tf.float32,
            shape=[],
            name=name + "_sigma"
        )

        # Target
        self.target_var = tf.placeholder(
            tf.int32,
            shape=[self.batch_size, ],
            name=name + "_targets"
        )

    def _conv1d_layer(self, input_var, filter_size, n_filters, stride, wd=0, activation='relu',bn=True):
        input_shape = input_var.get_shape()
        n_batches = input_shape[0].value
        input_dims = input_shape[1].value
        n_in_filters = input_shape[3].value
        name = "l{}_conv".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            output = conv_1d(name="conv1d", input_var=input_var, filter_shape=[filter_size, 1, n_in_filters, n_filters],
                             stride=stride, bias=None, wd=wd)

            # # MONITORING
            # self.monitor_vars.append(("{}_before_bn".format(name), output))
            if bn:
                output = batch_norm_new(name="bn", input_var=output, is_train=self.is_train)

            # # MONITORING
            # self.monitor_vars.append(("{}_after_bn".format(name), output))

            # output = leaky_relu(name="leaky_relu", input_var=output)
            if activation=='relu':
                output = tf.nn.relu(output, name="relu")
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def build_model(self, input_var, sigma):
        # add noise

        #if self.is_train:
        #    input_var = input_var + tf.random_normal(shape=tf.shape(input_var), stddev=sigma)
        #else:
        #    input_var = input_var + tf.random_normal(shape=tf.shape(input_var), stddev=sigma)

        # architecture
        for l in range(len(self.featuremap_sizes)):
            c = self._conv1d_layer(input_var=input_var, filter_size=self.filter_sizes[l], n_filters=self.featuremap_sizes[l], stride=self.strides[l])
        latent_vec = c
        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        c = flatten(name=name, input_var=c)
        self.activations.append((name, c))
        self.layer_idx += 1 
        
        name = "fc".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            output_tmp = fc(name="name", input_var=c, n_hiddens=256, bias=None, wd=0)
            output_tmp = batch_norm_new(name="bn", input_var=output_tmp, is_train=self.is_train)
            output_temp = tf.nn.relu(output_tmp, name="relu")
        self.activations.append((name, output_temp))
        
        return latent_vec, output_temp

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.variable_scope(self.name) as scope:
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            latent_vec, network = self.build_model(input_var=self.input_var, sigma=self.sigma)

            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1
            # Outputs of softmax linear are logits
            self.logits = network
            
            ######### Compute loss #########
            # denoising loss
            # denoising_loss = tf.nn.l2_loss(denoised_output - self.input_var)
            # denoising_loss = tf.reduce_mean(denoising_loss, name="denoising_loss")

            # Cross-entropy loss
            staging_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.target_var,
                name="sparse_softmax_cross_entropy_with_logits"
            )
            staging_loss = tf.reduce_mean(staging_loss, name="staging_loss")

            # Regularization loss
            regular_loss = tf.add_n(
                tf.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )
            
            # Total loss
            self.loss_op = tf.add(staging_loss, regular_loss)
            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)

            # denoised_output
            self.denoised_op = self.input_var


class ConvDecNet(ConvEncNet):

    def __init__(
        self,
        batch_size,
        input_dims, 
        n_classes, 
        is_train,
        reuse_params, 
        use_dropout, 
        name="convDecNet"
    ):
        super(self.__class__, self).__init__(
            batch_size,
            input_dims,
            n_classes,
            is_train,
            reuse_params,
            use_dropout,
            name=name
        )
        self.filter_sizes = self.filter_sizes[::-1]
        self.strides = self.strides[::-1]
        self.featuremap_sizes = self.featuremap_sizes[::-1]
        self.featuremap_sizes = self.featuremap_sizes[1:]
        self.featuremap_sizes.append(1)
        
    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.placeholder(
            tf.float32, 
            shape=[self.batch_size, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        #sigma
        self.sigma = tf.placeholder(
            tf.float32,
            shape=[],
            name=name + "_sigma"
        )
        # Target
        self.target_var = tf.placeholder(
            tf.int32, 
            shape=[self.batch_size, ],
            name=name + "_targets"
        )
    def _deconv1d_layer(self, input_var, filter_size, n_filters, stride, wd=0, activation='relu',bn=True):
        input_shape = input_var.get_shape()
        n_in_filters = input_shape[3].value
        name = "l{}_conv".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            output = deconv_1d(name="conv1d", input_var=input_var, filter_shape=[filter_size, 1, n_filters, n_in_filters],
                             stride=stride, bias=None, wd=wd)

            # # MONITORING
            # self.monitor_vars.append(("{}_before_bn".format(name), output))
            if bn:
                output = batch_norm_new(name="bn", input_var=output, is_train=self.is_train)

            # # MONITORING
            # self.monitor_vars.append(("{}_after_bn".format(name), output))

            # output = leaky_relu(name="leaky_relu", input_var=output)
            if activation=='relu':
                output = tf.nn.relu(output, name="relu")
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def build_model(self, input_var, sigma):
        # List to store the output of each CNNs
        output_conns = []

        # denoising network
        latent_vec, network = super(self.__class__, self).build_model(
            input_var=input_var, sigma=self.sigma
        )
        # Softmax linear
        name = "l{}_softmax_linear".format(self.layer_idx)
        network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
        self.activations.append((name, network))

        for l in range(len(self.featuremap_sizes)):
            c = self._deconv1d_layer(input_var=latent_vec, filter_size=self.filter_sizes[l],
                                   n_filters=self.featuremap_sizes[l], stride=self.strides[l])
        return c, network 
    
    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.variable_scope(self.name) as scope:
            
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()
            # Build model
            denoised_output, network = self.build_model(input_var=self.input_var, sigma=self.sigma)

            self.layer_idx += 1
            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########

            # denoising loss
            denoising_loss = tf.nn.l2_loss(denoised_output - self.input_var)
            denoising_loss = tf.reduce_mean(denoising_loss,  name="denoising_loss")

            # Cross-entropy loss
            staging_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.target_var,
                name="sparse_softmax_cross_entropy_with_logits"
            )
            staging_loss = tf.reduce_mean(staging_loss, name="staging_loss")

            # Regularization loss
            regular_loss = tf.add_n(
                tf.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )


            # Total loss
            self.loss_op = tf.add(staging_loss, regular_loss)
            self.loss_op =tf.add(tf.add(denoising_loss, staging_loss), regular_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)

            #denoised_output
            self.denoised_op = denoised_output

