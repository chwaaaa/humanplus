import itertools
import os
import re
import time

from datetime import datetime
import pdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score

from deepsleep.data_loader import NonSeqDataLoader, SeqDataLoader
from deepsleep.decoding_model_DA_l1 import DecodingDeepSleepNet, DecodingDeepFeatureNet
from deepsleep.decoding_model_perm import DeepFeatureNet
# , DecodingDeepSleepNet

from deepsleep.optimize import adam, adam_clipping_list_lr
from deepsleep.utils import iterate_minibatches, iterate_batch_seq_minibatches

# from tensorlayer.db import TensorDB
# from tensorlayer.db import JobStatus

import pdb


def cal_psnr(im1, im2):
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    maximum = np.max(im1)
    psnr = 10 * np.log10(maximum ** 2 / mse)
    return psnr


def pgd_l2_untargeted_mostlikely(model, X, epsilon=1.0, alpha=0.05, num_iter=20, randomize=False):
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
    else:
        delta = torch.zeros_like(X, requires_grad=True)
    delta = torch.zeros_like(X, requires_grad=True)

    yp = model(X)
    y = yp.max(dim=1)[1]

    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()

    return delta.detach()


def pgd_linf_untargeted_mostlikely_defense(model, sess, feed_dict, epsilon=0.2, stepsize=0.1, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    x = feed_dict[model.input_var]
    epsilon = np.max(x)
    if randomize:
        delta = np.random.uniform(low=-epsilon, high = epsilon, size=x.shape)
        delta = delta * 2 * epsilon - epsilon
    else:
        delta = np.zeros_like(x)
        
    for t in range(num_iter):
        feed_dict[model.delta] = delta
        delta_grad = sess.run(model.delta_grad, feed_dict=feed_dict)
        delta = np.clip((delta + stepsize * delta_grad), -epsilon, epsilon)
        #sess.run(model.delta_grad_stop_op, feed_dict = feed_dict)
    return delta


def pgd_linf_untargeted_mostlikely_attack(model, sess, feed_dict, epsilon=0.1, stepsize=0.1, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    x = feed_dict[model.input_var]
    epsilon = np.max(x)/2
    if randomize:
        delta = np.random.uniform(low=-epsilon, high=epsilon, size=x.shape)
        delta = delta * 2 * epsilon - epsilon
    else:
        delta = np.zeros_like(x)
    
    for t in range(num_iter):
        feed_dict[model.delta] = delta
        #print sess.run(model.adv_loss, feed_dict=feed_dict)
        delta_grad = sess.run(model.delta_grad, feed_dict=feed_dict)
        delta = np.clip((delta + stepsize * delta_grad), -epsilon, epsilon)
        # sess.run(model.delta_grad_stop_op, feed_dict = feed_dict)
    return delta

class Trainer(object):

    def __init__(
            self,
            interval_plot_filter=50,
            interval_save_model=100,
            interval_print_cm=10
    ):
        self.interval_plot_filter = interval_plot_filter
        self.interval_save_model = interval_save_model
        self.interval_print_cm = interval_print_cm

    def print_performance(self, sess, output_dir, network_name,
                          n_train_examples, n_valid_examples,
                          train_cm, valid_cm, epoch, n_epochs,
                          train_duration, train_loss_st, train_loss_de, train_acc, train_f1, train_psnr,
                          valid_duration, valid_loss_st, valid_loss_de, valid_acc, valid_f1, valid_psnr):
        # Get regularization loss
        train_reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
        train_reg_loss_value = sess.run(train_reg_loss)
        valid_reg_loss_value = train_reg_loss_value

        # Print performance
        if ((epoch + 1) % self.interval_print_cm == 0) or ((epoch + 1) == n_epochs):
            print " "
            print "[{}] epoch {}:".format(
                datetime.now(), epoch + 1
            )
            print (
                "train ({:.3f} sec): n={}, st_loss={:.3f} de_loss={:.3f} ({:.3f}), acc={:.3f}, "
                "f1={:.3f}, psnr={:.3f}".format(
                    train_duration, n_train_examples,
                    train_loss_st, train_loss_de, train_reg_loss_value,
                    train_acc, train_f1, train_psnr
                )
            )
            print train_cm
            print (
                "valid ({:.3f} sec): n={}, st_loss={:.3f} de_loss={:.3f} ({:.3f}), acc={:.3f}, "
                "f1={:.3f}, psnr={:.3f}".format(
                    valid_duration, n_valid_examples,
                    valid_loss_st, valid_loss_de, valid_reg_loss_value,
                    valid_acc, valid_f1, valid_psnr
                )
            )
            print valid_cm
            print " "
        else:
            print (
                "epoch {}: "
                "train ({:.2f} sec): n={}, st_loss={:.3f} de_loss={:.3f} ({:.3f}), "
                "acc={:.3f}, f1={:.3f}, psnr={:.3f} | "
                "valid ({:.2f} sec): n={}, st_loss={:.3f} de_loss={:.3f} ({:.3f}), "
                "acc={:.3f}, f1={:.3f}, psnr={:.3f}".format(
                    epoch + 1,
                    train_duration, n_train_examples,
                    train_loss_st, train_loss_de, train_reg_loss_value,
                    train_acc, train_f1, train_psnr,
                    valid_duration, n_valid_examples,
                    valid_loss_st, valid_loss_de, valid_reg_loss_value,
                    valid_acc, valid_f1, valid_psnr
                )
            )

    def print_denoising_performance(self, sess, output_dir, network_name,
                                    n_train_examples, n_valid_examples,
                                    epoch, n_epochs,
                                    train_duration, train_loss, train_psnr,
                                    valid_duration, valid_loss, valid_psnr):
        # Get regularization loss
        train_reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
        train_reg_loss_value = sess.run(train_reg_loss)
        valid_reg_loss_value = train_reg_loss_value

        # Print performance
        if ((epoch + 1) % self.interval_print_cm == 0) or ((epoch + 1) == n_epochs):
            print " "
            print "[{}] epoch {}:".format(
                datetime.now(), epoch + 1
            )
            print (
                "[{}] train ({:.3f} sec): n={}, loss={:.3f} ({:.3f}), psnr={:.3f}".format(
                    network_name, train_duration, n_train_examples,
                    train_loss, train_reg_loss_value,
                    train_psnr
                )
            )
            print (
                "[{}] valid ({:.3f} sec): n={}, loss={:.3f} ({:.3f}), psnr={:.3f}".format(
                    network_name, valid_duration, n_valid_examples,
                    valid_loss, valid_reg_loss_value,
                    valid_psnr
                )
            )
            print " "
        else:
            print (
                "[{}] epoch {}: "
                "train ({:.2f} sec): n={}, loss={:.3f} ({:.3f}), "
                "psnr={:.3f} | "
                "valid ({:.2f} sec): n={}, loss={:.3f} ({:.3f}), "
                "psnr={:.3f}".format(
                    network_name, epoch + 1,
                    train_duration, n_train_examples,
                    train_loss, train_reg_loss_value,
                    train_psnr,
                    valid_duration, n_valid_examples,
                    valid_loss, valid_reg_loss_value,
                    valid_psnr
                )
            )

    def print_network(self, network):
        print "inputs ({}): {}".format(
            network.inputs.name, network.inputs.get_shape()
        )
        print "targets ({}): {}".format(
            network.targets.name, network.targets.get_shape()
        )
        for name, act in network.activations:
            print "{} ({}): {}".format(name, act.name, act.get_shape())
        print " "

    def plot_filters(self, sess, epoch, reg_exp, output_dir, n_viz_filters):
        conv_weight = re.compile(reg_exp)
        for v in tf.trainable_variables():
            value = sess.run(v)
            if conv_weight.match(v.name):
                weights = np.squeeze(value)
                # Only plot conv that has one channel
                if len(weights.shape) > 2:
                    continue
                weights = weights.T
                plt.figure(figsize=(18, 10))
                plt.title(v.name)
                for w_idx in xrange(n_viz_filters):
                    plt.subplot(4, 4, w_idx + 1)
                    plt.plot(weights[w_idx])
                    plt.axis("tight")
                plt.savefig(os.path.join(
                    output_dir, "{}_{}.png".format(
                        v.name.replace("/", "_").replace(":0", ""),
                        epoch + 1
                    )
                ))
                plt.close("all")


class DecodingDeepFeatureNetTrainer(Trainer):

    def __init__(
            self,
            data_dir,
            output_dir,
            n_folds,
            fold_idx,
            batch_size,
            input_dims,
            n_classes,
            interval_plot_filter=50,
            interval_save_model=100,
            interval_print_cm=10
    ):
        super(self.__class__, self).__init__(
            interval_plot_filter=interval_plot_filter,
            interval_save_model=interval_save_model,
            interval_print_cm=interval_print_cm
        )

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes

    def _run_epoch(self, sess, epoch, network, source_network, inputs, targets, train_op, is_train):
        start_time = time.time()
        y = []
        y_true = []
        total_loss_st, total_loss_de, total_psnr, n_batches = 0.0, 0.0, 0.0, 0
        is_shuffle = True if is_train else False

        for x_batch, y_batch in iterate_minibatches(inputs,
                                                    targets,
                                                    self.batch_size,
                                                    shuffle=is_shuffle):
            
            
            # encoder for clean signal
            sigma = 0.0
            feed_dict = {
                network.input_var: x_batch,
                network.sigma: 0.0 ,
                network.target_var: y_batch
            }
            source_feed_dict = {
                source_network.input_var: x_batch,
                source_network.sigma: sigma,
                source_network.target_var: y_batch
            }
            if is_train:
                perm_delta = pgd_linf_untargeted_mostlikely_defense(source_network, sess, source_feed_dict)
                feed_dict[network.input_var] = x_batch + perm_delta
            else:
                perm_delta = pgd_linf_untargeted_mostlikely_attack(source_network, sess, source_feed_dict)
                feed_dict[network.input_var] = x_batch + perm_delta
                
            _, loss_st, y_pred = sess.run(
                [train_op[0], network.staging_loss, network.pred_op],
                feed_dict=feed_dict
            )
        

            # decoder for noisy signal
            feed_dict[network.sigma] = np.random.random()  * 8.0

            _, loss_de, x_pred = sess.run(
                [train_op[1], network.denoising_loss, network.denoised_op],
                feed_dict=feed_dict
            )
            psnr = cal_psnr(x_batch, x_pred)

            total_loss_st += loss_st
            total_loss_de += loss_de
            total_psnr += psnr
            n_batches += 1
            y.append(y_pred)
            y_true.append(y_batch)

            # Check the loss value
            assert not np.isnan(loss_de), \
                "Model diverged with loss = NaN"

        duration = time.time() - start_time
        total_loss_st /= n_batches
        total_loss_de /= n_batches

        total_psnr /= n_batches

        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)

        return total_y_true, total_y_pred, total_loss_st, total_loss_de, total_psnr, duration

    def train(self, source_model_path, n_epochs, resume):
        with tf.Graph().as_default(), tf.Session() as sess:
            # Build training and validation networks
            train_net = DecodingDeepFeatureNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=True,
                reuse_params=False,
                use_dropout=True
            )
            valid_net = DecodingDeepFeatureNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=False,
                reuse_params=True,
                use_dropout=True
            )
            source_net = DeepFeatureNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=False,
                reuse_params=False,
                use_dropout=True,
                name = "source_deepfeaturenet"
            )
            
            # Initialize parameters
            source_net.init_ops()
            train_net.init_ops()
            valid_net.init_ops()

            print "Network (layers={})".format(len(train_net.activations))
            print "inputs ({}): {}".format(
                train_net.input_var.name, train_net.input_var.get_shape()
            )
            print "targets ({}): {}".format(
                train_net.target_var.name, train_net.target_var.get_shape()
            )
            for name, act in train_net.activations:
                print "{} ({}): {}".format(name, act.name, act.get_shape())
            print " "

            source_model_name = 'deepfeaturenet'
            # Get list of all pretrained parameters for source network
            with np.load(source_model_path) as f:
                source_params = f.keys()
            # Remove the network-name-prefix
            for i in range(len(source_params)):
                source_params[i] = source_params[i].replace(source_model_name, "network")

            source_vars = [var for var in tf.trainable_variables() if var.name.startswith(source_net.name)]
            target_vars = [var for var in tf.trainable_variables() if var not in source_vars]

            # vars for denoising
            denoising_vars = target_vars[24:42]
            # vars for staging
            staging_vars = [var for var in tf.trainable_variables() if var not in denoising_vars]

            # Define optimization operations
            train_op1, grads_and_vars_op1 = adam(
                loss= train_net.staging_loss,
                lr=1e-4,
                train_vars=staging_vars
            )
            denoising_vars = target_vars[:-2]

            train_op2, grads_and_vars_op2 = adam(
                loss=0.3 * train_net.denoising_loss,
                lr=1e-3,
                train_vars=denoising_vars
            )

            # Make subdirectory for pretraining
            output_dir = os.path.join(self.output_dir, "fold{}".format(self.fold_idx), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Global step for resume training
            with tf.variable_scope(train_net.name) as scope:
                global_step = tf.Variable(0, name="global_step", trainable=False)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

            # Initialize variables in the graph
            sess.run(tf.global_variables_initializer())

            # Add the graph structure into the Tensorboard writer
            train_summary_wrt = tf.summary.FileWriter(
                os.path.join(output_dir, "train_summary"),
                sess.graph
            )
            # Load source model
            with np.load(source_model_path) as f:
                for k, v in f.iteritems():
                    if "Adam" in k or "softmax" in k or "power" in k or "global_step" in k:
                        continue
                    prev_k = k
                    k = k.replace(source_model_name, source_net.name)
                    tmp_tensor = tf.get_default_graph().get_tensor_by_name(k)
                    sess.run(
                        tf.assign(
                            tmp_tensor,
                            v
                        )
                    )
                    print "assigned {}: {} to {}: {}".format(
                        prev_k, v.shape, k, tmp_tensor.get_shape()
                    )
                    # Load pre-trained source model
                    print "Loading pre-trained parameters to the source model ..."
                    print " | --> {} from {}".format(source_model_name, source_model_path)

            # Resume the training if applicable
            if resume:
                if os.path.exists(output_dir):
                    if os.path.isfile(os.path.join(output_dir, "checkpoint")):
                        # Restore the last checkpoint
                        try:
                            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                        except:
                            output_dir = os.path.join(output_dir, "checkpoint")
                            saver.restore(sess, output_dir)
                        print "Model restored"
                        print "[{}] Resume pre-training ...\n".format(datetime.now())
                    else:
                        print "[{}] Start pre-training ...\n".format(datetime.now())
            else:
                print "[{}] Start pre-training ...\n".format(datetime.now())
            
            # Load data
            if sess.run(global_step) < n_epochs:
                data_loader = NonSeqDataLoader(
                    data_dir=self.data_dir,
                    n_folds=self.n_folds,
                    fold_idx=self.fold_idx
                )
                x_train, y_train, x_valid, y_valid = data_loader.load_train_data()

                # Performance history
                all_train_loss_st = np.zeros(n_epochs)
                all_train_loss_de = np.zeros(n_epochs)
                all_train_acc = np.zeros(n_epochs)
                all_train_f1 = np.zeros(n_epochs)
                all_train_psnr = np.zeros(n_epochs)
                all_valid_loss_st = np.zeros(n_epochs)
                all_valid_loss_de = np.zeros(n_epochs)
                all_valid_acc = np.zeros(n_epochs)
                all_valid_f1 = np.zeros(n_epochs)
                all_valid_psnr = np.zeros(n_epochs)

            # Loop each epoch
            for epoch in xrange(sess.run(global_step), n_epochs):
                y_true_train, y_pred_train, train_loss_st, train_loss_de, train_psnr, train_duration = \
                    self._run_epoch(
                        sess=sess, network=train_net, source_network = source_net, epoch=epoch,
                        inputs=x_train, targets=y_train,
                        train_op=[train_op1, train_op2],
                        is_train=True
                    )
                n_train_examples = len(y_true_train)
                train_cm = confusion_matrix(y_true_train, y_pred_train)
                train_acc = np.mean(y_true_train == y_pred_train)
                train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

                # Evaluate the model on the validation set
                y_true_val, y_pred_val, valid_loss_st, valid_loss_de, valid_psnr, valid_duration = \
                    self._run_epoch(
                        sess=sess, network=valid_net, source_network = source_net, epoch=epoch,
                        inputs=x_valid, targets=y_valid,
                        train_op=[tf.no_op(), tf.no_op()],
                        is_train=False
                    )
                n_valid_examples = len(y_true_val)
                valid_cm = confusion_matrix(y_true_val, y_pred_val)
                valid_acc = np.mean(y_true_val == y_pred_val)
                valid_f1 = f1_score(y_true_val, y_pred_val, average="macro")

                all_train_loss_st[epoch] = train_loss_st
                all_train_loss_de[epoch] = train_loss_de

                all_train_acc[epoch] = train_acc
                all_train_f1[epoch] = train_f1
                all_train_psnr[epoch] = train_psnr

                all_valid_loss_st[epoch] = valid_loss_st
                all_valid_loss_de[epoch] = valid_loss_de
                all_valid_acc[epoch] = valid_acc
                all_valid_f1[epoch] = valid_f1
                all_valid_psnr[epoch] = valid_psnr

                # Report performance
                self.print_performance(
                    sess, output_dir, train_net.name,
                    n_train_examples, n_valid_examples,
                    train_cm, valid_cm, epoch, n_epochs,
                    train_duration, train_loss_st, train_loss_de, train_acc, train_f1, train_psnr,
                    valid_duration, valid_loss_st, valid_loss_de, valid_acc, valid_f1, valid_psnr
                )

                # Save performance history
                np.savez(
                    os.path.join(output_dir, "perf_fold{}.npz".format(self.fold_idx)),
                    train_loss_st=all_train_loss_st, train_loss_de=all_train_loss_de,
                    valid_loss_st=all_valid_loss_st, valid_loss_de=all_valid_loss_de,
                    train_acc=all_train_acc, valid_acc=all_valid_acc,
                    train_f1=all_train_f1, valid_f1=all_valid_f1,
                    y_true_val=np.asarray(y_true_val),
                    y_pred_val=np.asarray(y_pred_val)
                )

                # Visualize weights from convolutional layers
                if ((epoch + 1) % self.interval_plot_filter == 0) or ((epoch + 1) == n_epochs):
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?\/l[0-9]+_conv\/(weights)", output_dir,
                                      16)
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?/l[0-9]+_conv\/conv1d\/(weights)",
                                      output_dir, 16)

                # Save checkpoint
                sess.run(tf.assign(global_step, epoch + 1))
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_path = os.path.join(
                        output_dir, "model_fold{}.ckpt".format(self.fold_idx)
                    )
                    saver.save(sess, save_path, global_step=global_step)
                    duration = time.time() - start_time
                    print "Saved model checkpoint ({:.3f} sec)".format(duration)

                # Save paramaters
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_dict = {}
                    for v in tf.global_variables():
                        save_dict[v.name] = sess.run(v)
                    np.savez(
                        os.path.join(
                            output_dir,
                            "params_fold{}.npz".format(self.fold_idx)),
                        **save_dict
                    )
                    duration = time.time() - start_time
                    print "Saved trained parameters ({:.3f} sec)".format(duration)

        print "Finish pre-training"
        return os.path.join(output_dir, "params_fold{}.npz".format(self.fold_idx))


class DecodingDeepSleepNetTrainer(Trainer):
    def __init__(
            self,
            data_dir,
            output_dir,
            n_folds,
            fold_idx,
            batch_size,
            input_dims,
            n_classes,
            seq_length,
            n_rnn_layers,
            return_last,
            interval_plot_filter=50,
            interval_save_model=100,
            interval_print_cm=10
    ):
        super(self.__class__, self).__init__(
            interval_plot_filter=interval_plot_filter,
            interval_save_model=interval_save_model,
            interval_print_cm=interval_print_cm
        )

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.return_last = return_last

    def _run_epoch(self, sess, network, source_network, inputs, targets, train_op, is_train):
        start_time = time.time()
        y = []
        y_true = []
        total_loss_st, total_loss_de, total_psnr, n_batches = 0.0, 0.0, 0.0, 0
        for sub_idx, each_data in enumerate(itertools.izip(inputs, targets)):
            each_x, each_y = each_data

            # # Initialize state of LSTM - Unidirectional LSTM
            # state = sess.run(network.initial_state)

            # Initialize state of LSTM - Bidirectional LSTM
            fw_state = sess.run(network.fw_initial_state)
            bw_state = sess.run(network.bw_initial_state)

            for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                                  targets=each_y,
                                                                  batch_size=self.batch_size,
                                                                  seq_length=self.seq_length):

                sigma = 0.0
                source_feed_dict = {
                    source_network.input_var: x_batch,
                    source_network.sigma: sigma,
                    source_network.target_var: y_batch
                }
                
                feed_dict = {
                    network.input_var: x_batch,
                    network.sigma: 0.0,
                    network.target_var: y_batch
                }
                
                
                for i, (c, h) in enumerate(network.fw_initial_state):
                    feed_dict[c] = fw_state[i].c
                    feed_dict[h] = fw_state[i].h

                for i, (c, h) in enumerate(network.bw_initial_state):
                    feed_dict[c] = bw_state[i].c
                    feed_dict[h] = bw_state[i].h
                
                if is_train:
                    perm_delta = pgd_linf_untargeted_mostlikely_defense(source_network, sess, source_feed_dict)
                    feed_dict[network.input_var] = x_batch + perm_delta
                else:
                    perm_delta = pgd_linf_untargeted_mostlikely_attack(source_network, sess, source_feed_dict)
                    feed_dict[network.input_var] = x_batch + perm_delta

                _, _, loss_st, y_pred, x_pred, fw_state, bw_state = sess.run(
                    [train_op[0], train_op[1], network.staging_loss,
                     network.pred_op, network.denoised_op, network.fw_final_state, network.bw_final_state],
                    feed_dict=feed_dict
                )
                '''
                # decoder for noisy signal
                feed_dict[network.sigma] = np.random.random() * (8.0)

                _, loss_de, x_pred, fw_state, bw_state = sess.run(
                    [train_op[2], network.denoising_loss, network.denoised_op, network.fw_final_state,
                     network.bw_final_state],
                    feed_dict=feed_dict
                )

                psnr = cal_psnr(x_batch, x_pred)
                '''
                total_loss_st += loss_st
                #total_loss_de += loss_de

                n_batches += 1
                y.append(y_pred)
                y_true.append(y_batch)
                #total_psnr += psnr
                # Check the loss value
                assert not np.isnan(loss_st), \
                    "Model diverged with loss = NaN"

        duration = time.time() - start_time
        total_loss_st /= n_batches
        total_loss_de /= n_batches

        total_psnr /= n_batches
        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)

        return total_y_true, total_y_pred, total_loss_st, total_loss_de, total_psnr, duration

    def finetune(self, pretrained_model_path, source_model_path, n_epochs, resume):
        pretrained_model_name = "decodingdeepfeaturenet"

        with tf.Graph().as_default(), tf.Session() as sess:
            # Build training and validation networks
            train_net = DecodingDeepSleepNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                seq_length=self.seq_length,
                n_rnn_layers=self.n_rnn_layers,
                return_last=self.return_last,
                is_train=True,
                reuse_params=False,
                use_dropout_feature=True,
                use_dropout_sequence=True
            )
            valid_net = DecodingDeepSleepNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                seq_length=self.seq_length,
                n_rnn_layers=self.n_rnn_layers,
                return_last=self.return_last,
                is_train=False,
                reuse_params=True,
                use_dropout_feature=True,
                use_dropout_sequence=True
            )
            source_net = DeepFeatureNet(
                batch_size=self.batch_size*self.seq_length,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=False,
                reuse_params=False,
                use_dropout=True,
                name = "source_deepfeaturenet"
            )

            # Initialize parameters            
            source_net.init_ops()
            train_net.init_ops()
            valid_net.init_ops()

            print "Network (layers={})".format(len(train_net.activations))
            print "inputs ({}): {}".format(
                train_net.input_var.name, train_net.input_var.get_shape()
            )
            print "targets ({}): {}".format(
                train_net.target_var.name, train_net.target_var.get_shape()
            )
            for name, act in train_net.activations:
                print "{} ({}): {}".format(name, act.name, act.get_shape())
            print " "

            # Get list of all pretrained parameters
            with np.load(pretrained_model_path) as f:
                pretrain_params = f.keys()
            # Remove the network-name-prefix
            for i in range(len(pretrain_params)):
                pretrain_params[i] = pretrain_params[i].replace(pretrained_model_name, "network")
            
            source_model_name = 'deepfeaturenet'
            # Get list of all pretrained parameters for source network
            with np.load(source_model_path) as f:
                source_params = f.keys()
            # Remove the network-name-prefix
            for i in range(len(source_params)):
                source_params[i] = source_params[i].replace(source_model_name, "network")
                
            # Get trainable variables of the pretrained, and new ones
            source_vars = [var for var in tf.trainable_variables() if var.name.startswith(source_net.name)]
            target_vars = [var for var in tf.trainable_variables() if var not in source_vars]
            train_vars1 = [v for v in target_vars
                           if v.name.replace(train_net.name, "network") in pretrain_params]
            # vars for denoising
            denoising_vars = train_vars1[24:42]
            
            # vars for staging
            staging_vars1 = [var for var in train_vars1 if var not in denoising_vars]
            staging_vars2 = list(set(target_vars) - set(train_vars1))
            
            train_op, grads_and_vars_op = adam(
                loss= train_net.staging_loss,
                lr=1e-6,
                train_vars=staging_vars1
            )

            train_op1, grads_and_vars_op1 = adam(
                loss= train_net.staging_loss,
                lr=1e-4,
                train_vars=staging_vars2
            )
            
            train_op2, grads_and_vars_op2 = adam(
                loss= 0.4 * train_net.denoising_loss,
                lr=1e-5,
                train_vars=target_vars[:-2]
            )

            # Make subdirectory for pretraining
            output_dir = os.path.join(self.output_dir, "fold{}".format(self.fold_idx), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Global step for resume training
            with tf.variable_scope(train_net.name) as scope:
                global_step = tf.Variable(0, name="global_step", trainable=False)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

            # Initialize variables in the graph
            sess.run(tf.global_variables_initializer())

            # Add the graph structure into the Tensorboard writer
            train_summary_wrt = tf.summary.FileWriter(
                os.path.join(output_dir, "train_summary"),
                sess.graph
            )
            load_pretrain=False
            # Resume the training if applicable
            if resume:
                if os.path.exists(output_dir):
                    if os.path.isfile(os.path.join(output_dir, "checkpoint")):
                        # Restore the last checkpoint
                        saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                        print "Model restored"
                        print "[{}] Resume fine-tuning ...\n".format(datetime.now())
                    else:
                        load_pretrain = True
            else:
                load_pretrain = True

            if load_pretrain:
                # Load pre-trained model
                print "Loading pre-trained parameters to the model ..."
                print " | --> {} from {}".format(pretrained_model_name, pretrained_model_path)
                with np.load(pretrained_model_path) as f:
                    for k, v in f.iteritems():
                        if "Adam" in k or "softmax" in k or "power" in k or "global_step" in k:
                            continue
                        prev_k = k
                        k = k.replace(pretrained_model_name, train_net.name)
                        tmp_tensor = tf.get_default_graph().get_tensor_by_name(k)
                        sess.run(
                            tf.assign(
                                tmp_tensor,
                                v
                            )
                        )
                        print "assigned {}: {} to {}: {}".format(
                            prev_k, v.shape, k, tmp_tensor.get_shape()
                        )

                        # Load pre-trained model
                        print "Loading pre-trained parameters to the target model ..."
                        print " | --> {} from {}".format(pretrained_model_name, pretrained_model_path)

                # Load source model
                with np.load(source_model_path) as f:
                    for k, v in f.iteritems():
                        if "Adam" in k or "softmax" in k or "power" in k or "global_step" in k:
                            if 'softmax' in k and 'Adam' not in k:
                                pass
                            else:
                                continue
                        prev_k = k
                        k = k.replace(source_model_name, source_net.name)
                        tmp_tensor = tf.get_default_graph().get_tensor_by_name(k)
                        sess.run(
                            tf.assign(
                                tmp_tensor,
                                v
                            )
                        )
                        print "assigned {}: {} to {}: {}".format(
                            prev_k, v.shape, k, tmp_tensor.get_shape()
                        )
                        # Load pre-trained source model
                        print "Loading pre-trained parameters to the source model ..."
                        print " | --> {} from {}".format(source_model_name, source_model_path)

                print " "
                print "[{}] Start fine-tuning ...\n".format(datetime.now())

            # Load data
            if sess.run(global_step) < n_epochs:
                data_loader = SeqDataLoader(
                    data_dir=self.data_dir,
                    n_folds=self.n_folds,
                    fold_idx=self.fold_idx
                )
                x_train, y_train, x_valid, y_valid = data_loader.load_train_data()

                # Performance history
                all_train_loss_st = np.zeros(n_epochs)
                all_train_loss_de = np.zeros(n_epochs)
                all_train_acc = np.zeros(n_epochs)
                all_train_f1 = np.zeros(n_epochs)
                all_train_psnr = np.zeros(n_epochs)

                all_valid_loss_st = np.zeros(n_epochs)
                all_valid_loss_de = np.zeros(n_epochs)
                all_valid_acc = np.zeros(n_epochs)
                all_valid_f1 = np.zeros(n_epochs)
                all_valid_psnr = np.zeros(n_epochs)

            # Loop each epoch
            for epoch in xrange(sess.run(global_step), n_epochs):
                # Update parameters and compute loss of training set
                y_true_train, y_pred_train, train_loss_st, train_loss_de, train_psnr, train_duration = \
                    self._run_epoch(
                        sess=sess, network=train_net, source_network=source_net,
                        inputs=x_train, targets=y_train,
                        train_op=[train_op, train_op1, train_op2],
                        is_train=True
                    )
                n_train_examples = len(y_true_train)
                train_cm = confusion_matrix(y_true_train, y_pred_train)
                train_acc = np.mean(y_true_train == y_pred_train)
                train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

                # Evaluate the model on the validation set
                y_true_val, y_pred_val, valid_loss_st, valid_loss_de, valid_psnr, valid_duration = \
                    self._run_epoch(
                        sess=sess, network=valid_net, source_network = source_net,
                        inputs=x_valid, targets=y_valid,
                        train_op=[tf.no_op(), tf.no_op(), tf.no_op()],
                        is_train=False
                    )
                n_valid_examples = len(y_true_val)
                valid_cm = confusion_matrix(y_true_val, y_pred_val)
                valid_acc = np.mean(y_true_val == y_pred_val)
                valid_f1 = f1_score(y_true_val, y_pred_val, average="macro")

                all_train_loss_st[epoch] = train_loss_st
                all_train_loss_de[epoch] = train_loss_de
                all_train_acc[epoch] = train_acc
                all_train_f1[epoch] = train_f1
                all_train_psnr[epoch] = train_psnr

                all_valid_loss_st[epoch] = valid_loss_st
                all_valid_loss_de[epoch] = valid_loss_de
                all_valid_acc[epoch] = valid_acc
                all_valid_f1[epoch] = valid_f1
                all_valid_psnr[epoch] = valid_psnr

                self.print_performance(
                    sess, output_dir, train_net.name,
                    n_train_examples, n_valid_examples,
                    train_cm, valid_cm, epoch, n_epochs,
                    train_duration, train_loss_st, train_loss_de, train_acc, train_f1, train_psnr,
                    valid_duration, valid_loss_st, valid_loss_de, valid_acc, valid_f1, valid_psnr
                )

                # Save performance history
                np.savez(
                    os.path.join(output_dir, "perf_fold{}.npz".format(self.fold_idx)),
                    train_loss_st=all_train_loss_st, train_loss_de=all_train_loss_de,
                    valid_loss_st=all_valid_loss_st, valid_loss_de=all_valid_loss_de,
                    train_acc=all_train_acc, valid_acc=all_valid_acc,
                    train_f1=all_train_f1, valid_f1=all_valid_f1,
                    train_psnr=all_train_psnr, valid_psnr=all_valid_psnr,

                    y_true_val=np.asarray(y_true_val),
                    y_pred_val=np.asarray(y_pred_val)
                )

                # Visualize weights from convolutional layers
                if ((epoch + 1) % self.interval_plot_filter == 0) or ((epoch + 1) == n_epochs):
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?\/l[0-9]+_conv\/(weights)", output_dir,
                                      16)
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?/l[0-9]+_conv\/conv1d\/(weights)",
                                      output_dir, 16)

                # Save checkpoint
                sess.run(tf.assign(global_step, epoch + 1))
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_path = os.path.join(
                        output_dir, "model_fold{}.ckpt".format(self.fold_idx)
                    )
                    saver.save(sess, save_path, global_step=global_step)
                    duration = time.time() - start_time
                    print "Saved model checkpoint ({:.3f} sec)".format(duration)

                # Save paramaters
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_dict = {}
                    for v in tf.global_variables():
                        save_dict[v.name] = sess.run(v)
                    np.savez(
                        os.path.join(
                            output_dir,
                            "params_fold{}.npz".format(self.fold_idx)),
                        **save_dict
                    )
                    duration = time.time() - start_time
                    print "Saved trained parameters ({:.3f} sec)".format(duration)

        print "Finish training"
        return os.path.join(output_dir, "params_fold{}.npz".format(self.fold_idx))
