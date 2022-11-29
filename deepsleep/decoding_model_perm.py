import tensorflow as tf
import numpy as np
from deepsleep.nn import *
from sklearn.metrics import confusion_matrix, f1_score

from random import *
import pdb


class DeepFeatureNet(object):

    def __init__(
            self,
            batch_size,
            input_dims,
            n_classes,
            is_train,
            reuse_params,
            use_dropout,
            name="deepfeaturenet"
    ):
        self.batch_size = batch_size
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
        self.delta = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.input_dims, 1, 1],
            name=name + "_delta"
        )
        # sigma
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

    def _conv1d_layer(self, input_var, filter_size, n_filters, stride, wd=0):

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

            output = batch_norm_new(name="bn", input_var=output, is_train=self.is_train)

            # # MONITORING
            # self.monitor_vars.append(("{}_after_bn".format(name), output))

            # output = leaky_relu(name="leaky_relu", input_var=output)
            output = tf.nn.relu(output, name="relu")
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def build_model(self, input_var):
        # List to store the output of each CNNs
        output_conns = []
        #input_var = input_var + tf.random_normal(shape=tf.shape(input_var), stddev=self.sigma)

        ######### CNNs with small filter size at the first layer #########
        # Convolution
        # network = self._conv1d_layer(input_var=input_var, filter_size=128, n_filters=64, stride=16, wd=1e-3)
        network = self._conv1d_layer(input_var=input_var, filter_size=50, n_filters=64, stride=6,
                                     wd=1e-3)  # (, 500, 1, 64)
        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=8, stride=8)  # (, 63, 1, 64)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1
        latent_vec = network

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### CNNs with large filter size at the first layer #########

        # Convolution
        # network = self._conv1d_layer(input_var=input_var, filter_size=1024, n_filters=64, stride=128)
        network = self._conv1d_layer(input_var=input_var, filter_size=400, n_filters=64, stride=50)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=2, stride=2)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### Aggregate and link two CNNs #########
        # Concat
        name = "l{}_concat".format(self.layer_idx)
        network = tf.concat(output_conns, 1, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        return latent_vec, network

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.variable_scope(self.name) as scope:
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            latent_vec, network = self.build_model(input_var=self.input_var)
            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1
            
            # build model for permuted input
            self.layer_idx = 1
            scope.reuse_variables()
            latent_vec_pm, network_pm = self.build_model(input_var=self.input_var + self.delta)
            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network_pm = fc(name=name, input_var=network_pm, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.layer_idx += 1
            self.network_pm = network_pm

            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########

            # Cross-entropy loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.target_var,
                name="sparse_softmax_cross_entropy_with_logits"
            )
            loss = tf.reduce_mean(loss, name="cross_entropy")

            # Regularization loss
            regular_loss = tf.add_n(
                tf.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )

            # Total loss
            self.loss_op = tf.add(loss, regular_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)
            
            # advarsarial loss            
            adv_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.network_pm,
                labels=self.pred_op,
                name="sparse_softmax_cross_entropy_with_logits"
            )
            
            self.adv_loss = tf.reduce_mean(adv_loss, name="cross_entropy")
            self.delta_grad = tf.math.sign(tf.gradients(self.adv_loss, self.delta)[0])
            self.delta_grad_stop_op = tf.stop_gradient(self.delta_grad)
            
class DecodingDeepFeatureNet(object):

    def __init__(
            self,
            batch_size,
            input_dims,
            n_classes,
            is_train,
            reuse_params,
            use_dropout,
            name="decodingdeepfeaturenet"
    ):
        self.batch_size = batch_size
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
        # sigma
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

    def _conv1d_layer(self, input_var, filter_size, n_filters, stride, wd=0):

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

            output = batch_norm_new(name="bn", input_var=output, is_train=self.is_train)

            output = tf.nn.relu(output, name="relu")
        self.activations.append((name, output))
        self.layer_idx += 1
        return output
    
    def _deconv1d_layer(self, input_var, filter_size, n_filters, stride, output_shape = None, wd=0, activation='relu',bn=True):
        input_shape = input_var.get_shape()
        n_in_filters = input_shape[3].value
        name = "l{}_deconv".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            output = deconv_1d(name="deconv1d", input_var=input_var, filter_shape=[filter_size, 1, n_filters, n_in_filters],
                             output_shape = output_shape, stride=stride, bias=None, wd=wd)

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

    def build_model(self, input_var):
        # List to store the output of each CNNs
        output_conns = []
        
        if not self.is_train:
            input_var = input_var + tf.random_normal(shape=tf.shape(input_var), stddev=self.sigma)

        ######### CNNs with small filter size at the first layer #########

        # Convolution
        # network = self._conv1d_layer(input_var=input_var, filter_size=128, n_filters=64, stride=16, wd=1e-3)
        network = self._conv1d_layer(input_var=input_var, filter_size=50, n_filters=64, stride=6, wd=1e-3)  # (, 500, 1, 64)
        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=8, stride=8)  # (, 63, 1, 64)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1
        # latent vector to decode
        latent_vec = network

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### CNNs with large filter size at the first layer #########

        # Convolution
        # network = self._conv1d_layer(input_var=input_var, filter_size=1024, n_filters=64, stride=128)
        network = self._conv1d_layer(input_var=input_var, filter_size=400, n_filters=64, stride=50)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=2, stride=2)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### Aggregate and link two CNNs #########
        # Concat
        name = "l{}_concat".format(self.layer_idx)
        network = tf.concat(output_conns, 1, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        ######### Decoder #########

        latent_vec = self._deconv1d_layer(input_var=latent_vec, filter_size=1, n_filters=128, stride=4, output_shape=63)

        # deconvolution
        latent_vec = self._deconv1d_layer(input_var=latent_vec, filter_size=8, n_filters=128, stride=1)
        latent_vec = self._deconv1d_layer(input_var=latent_vec, filter_size=8, n_filters=128, stride=1)
        latent_vec = self._deconv1d_layer(input_var=latent_vec, filter_size=8, n_filters=64, stride=1)

        # upsample
        latent_vec = self._deconv1d_layer(input_var=latent_vec, filter_size=1, n_filters=64, stride=8, output_shape=500)
        # deconvolution
        denoising_output = self._deconv1d_layer(input_var=latent_vec, filter_size=50, n_filters=1, stride=6)
        return denoising_output, network

    def init_ops(self):
        self._build_placeholder()
        # Get loss and prediction operations
        with tf.variable_scope(self.name) as scope:
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            denoised_output, network = self.build_model(input_var=self.input_var)

            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########
            # Cross-entropy loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.target_var,
                name="sparse_softmax_cross_entropy_with_logits"
            )
            staging_loss = tf.reduce_mean(loss, name="cross_entropy")
            
            # Regularization loss
            self.regular_loss = tf.add_n(
                tf.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
                )
            pdb.set_trace()
            # denoising loss
            # FIXME
            denoising_loss = tf.nn.l2_loss(denoised_output - self.input_var)
            denoising_loss = tf.reduce_mean(denoising_loss, name="denoising_loss")
            
            self.staging_loss = tf.reduce_sum(staging_loss) / self.batch_size + self.regular_loss

            self.denoising_loss = denoising_loss
            # Total loss
            self.loss_op =tf.add(0.1*self.denoising_loss, self.staging_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)
            # denoised_output
            self.denoised_op = denoised_output


class DecodingDeepSleepNet(DecodingDeepFeatureNet):

    def __init__(
            self,
            batch_size,
            input_dims,
            n_classes,
            seq_length,
            n_rnn_layers,
            return_last,
            is_train,
            reuse_params,
            use_dropout_feature,
            use_dropout_sequence,
            name="decodingdeepsleepnet"
    ):
        super(self.__class__, self).__init__(
            batch_size=batch_size,
            input_dims=input_dims,
            n_classes=n_classes,
            is_train=is_train,
            reuse_params=reuse_params,
            use_dropout=use_dropout_feature,
            name=name
        )

        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.return_last = return_last
        
        self.use_dropout_sequence = use_dropout_sequence

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.placeholder(
            tf.float32,
            shape=[self.batch_size * self.seq_length, self.input_dims, 1, 1],
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
            shape=[self.batch_size * self.seq_length, ],
        )

    def build_model(self, input_var):
        # Create a network with superclass method
        denoising_output, network = super(self.__class__, self).build_model(
            input_var=self.input_var
        )
        # Residual (or shortcut) connection
        output_conns = []

        # Fully-connected to select some part of the output to add with the output from bi-directional LSTM
        name = "l{}_fc".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            output_tmp = fc(name="fc", input_var=network, n_hiddens=1024, bias=None, wd=0)
            output_tmp = batch_norm_new(name="bn", input_var=output_tmp, is_train=self.is_train)
            # output_tmp = leaky_relu(name="leaky_relu", input_var=output_tmp)
            output_tmp = tf.nn.relu(output_tmp, name="relu")
        self.activations.append((name, output_tmp))
        self.layer_idx += 1
        output_conns.append(output_tmp)

        ######################################################################

        # Reshape the input from (batch_size * seq_length, input_dim) to
        # (batch_size, seq_length, input_dim)
        name = "l{}_reshape_seq".format(self.layer_idx)
        input_dim = network.get_shape()[-1].value
        seq_input = tf.reshape(network,
                               shape=[-1, self.seq_length, input_dim],
                               name=name)
        assert self.batch_size == seq_input.get_shape()[0].value
        self.activations.append((name, seq_input))
        self.layer_idx += 1

        # Bidirectional LSTM network
        name = "l{}_bi_lstm".format(self.layer_idx)
        hidden_size = 512  # will output 1024 (512 forward, 512 backward)

        def lstm_cell(hidden_size):
            lstm = tf.nn.rnn_cell.LSTMCell(hidden_size,
                                           use_peepholes=True,
                                           state_is_tuple=True)
            if self.use_dropout_sequence:
                keep_prob = 0.5 if self.is_train else 1.0
                drop = tf.nn.rnn_cell.DropoutWrapper(
                    lstm,
                    output_keep_prob=keep_prob
                )
            return drop

        with tf.variable_scope(name) as scope:

            fw_cell = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell(hidden_size) for _ in range(self.n_rnn_layers)],
                state_is_tuple=True)
            bw_cell = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell(hidden_size) for _ in range(self.n_rnn_layers)],
                state_is_tuple=True)

            # Initial state of RNN
            self.fw_initial_state = fw_cell.zero_state(self.batch_size, tf.float32)
            self.bw_initial_state = bw_cell.zero_state(self.batch_size, tf.float32)

            # Feedforward to MultiRNNCell
            list_rnn_inputs = tf.unstack(seq_input, axis=1)
            outputs, fw_state, bw_state = tf.nn.static_bidirectional_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=list_rnn_inputs,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state
            )

            if self.return_last:
                network = outputs[-1]
            else:
                network = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2],
                                     name=name)
            self.activations.append((name, network))
            self.layer_idx += 1

            self.fw_final_state = fw_state
            self.bw_final_state = bw_state

        # Append output
        output_conns.append(network)

        ######################################################################

        # Add
        name = "l{}_add".format(self.layer_idx)
        network = tf.add_n(output_conns, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout_sequence:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        return denoising_output, network

    def init_ops(self):
        self._build_placeholder()

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            denoised_output, network = self.build_model(input_var=self.input_var)

            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########
            # denoising loss
            denoising_loss = tf.nn.l2_loss(denoised_output - self.input_var)
            denoising_loss = tf.reduce_mean(denoising_loss,  name="denoising_loss")

            # Weighted cross-entropy loss for a sequence of logits (per example)
            staging_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [self.target_var],
                [tf.ones([self.batch_size * self.seq_length])],
                name="sequence_loss_by_example"
            )

            # Regularization loss
            self.regular_loss = tf.add_n(
                tf.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )
            
            self.staging_loss = tf.reduce_sum(staging_loss) / self.batch_size + self.regular_loss

            self.denoising_loss = denoising_loss
            # Total loss
            self.loss_op =tf.add(0.1*self.denoising_loss, self.staging_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)
            
            #denoised_output
            self.denoised_op = denoised_output

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = tf.reduce_mean((im1 - im2 ** 2))
    maximum=tf.reduce_max(im1)
    psnr = 10 * tf.math.log(maximum ** 2 / mse)/tf.log(tf.constant(10, dtype=maximum.dtype))
    return psnr

def cal_f1(y_pred, y_true):
    pdb.set_trace()
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    TP = tf.count_nonzero(y_pred * y_true, axis=None)
    FP = tf.count_nonzero(y_pred * (y_true - 1), axis=None)
    FN = tf.count_nonzero((y_pred - 1) * y_true, axis=None)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall) 
    return f1