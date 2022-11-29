import itertools
import os
import re
import time

from datetime import datetime

import matplotlib
from sklearn.utils import shuffle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
import scipy.signal as sps

from deepsleep.data_loader import NonSeqDataLoader, SeqDataLoader
#for unseen data
from deepsleep.mass_data_loader import NonSeqDataLoader as mass_NonSeqDataLoader
from deepsleep.mass_data_loader import SeqDataLoader as mass_SeqDataLoader

from deepsleep.model_DA_unsupervised import DeepFeatureNet, DeepSleepNet

from deepsleep.optimize import adam, adam_clipping_list_lr
from deepsleep.utils import iterate_minibatches, iterate_minibatches_subject, iterate_batch_seq_minibatches, iterate_batch_seq_minibatches_subject, get_balance_class_subject_by_target

# from tensorlayer.db import TensorDB
# from tensorlayer.db import JobStatus

import pdb

def cal_psnr(im1, im2):
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    maximum = np.max(im1)
    psnr = 10 * np.log10(maximum ** 2 / mse)
    return psnr


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

    def print_performance(self, sess, network_name,
                          src_n_train_examples, src_n_valid_examples, tar_n_train_examples, tar_n_valid_examples,
                          src_train_cm, src_valid_cm, tar_train_cm, tar_valid_cm, epoch, n_epochs,
                          train_duration, src_train_loss_st, src_train_loss_do, src_train_loss_cl, src_train_loss_sub, src_train_loss_syn, src_train_acc, src_train_f1,
                          tar_train_loss_st, tar_train_loss_do, tar_train_loss_cl,  tar_train_loss_sub, tar_train_loss_syn, tar_train_acc, tar_train_f1,
                          valid_duration, src_valid_loss_st, src_valid_loss_do, src_valid_loss_cl, src_valid_loss_sub, src_valid_loss_syn, src_valid_acc, src_valid_f1,
                          tar_valid_loss_st, tar_valid_loss_do, tar_valid_loss_cl, tar_valid_loss_sub, tar_valid_loss_syn, tar_valid_acc, tar_valid_f1,
                          src_train_n_batches, tar_train_n_batches, src_valid_n_batches, tar_valid_n_batches):
        # Get regularization loss
        #train_reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
        #train_reg_loss_value = sess.run(train_reg_loss)
        #valid_reg_loss_value = train_reg_loss_value

        # Print performance

        if ((epoch + 1) % self.interval_print_cm == 0) or ((epoch + 1) == n_epochs):
            print " "
            print "[{}] epoch {}:".format(
                datetime.now(), epoch + 1
            )
            print (
                "soruce train ({:.3f} sec): n={}, st_loss={:.3f} do_loss={:.3f}, sub_loss= {:.3f}, cl_loss={:.3f}, syn_loss={:.3f}, acc={:.3f}, "
                "f1={:.3f}".format(
                    train_duration, src_n_train_examples,
                    sum(src_train_loss_st)/src_train_n_batches, sum(src_train_loss_do)/src_train_n_batches, sum(src_train_loss_cl)/src_train_n_batches, sum(src_train_loss_sub)/src_train_n_batches, sum(src_train_loss_syn)/src_train_n_batches,
                    src_train_acc, src_train_f1
                )
            )
            print src_train_cm
            print (
                "target train ({:.3f} sec): n={}, st_loss={:.3f} do_loss={:.3f}, cl_loss={:.3f},  sub_loss= {:.3f}, csyn_loss={:.3f}, acc={:.3f}, "
                "f1={:.3f}".format(
                    train_duration, tar_n_train_examples,
                    sum(tar_train_loss_st)/tar_train_n_batches, sum(tar_train_loss_do)/tar_train_n_batches, sum(tar_train_loss_cl)/tar_train_n_batches,sum(tar_train_loss_sub)/tar_train_n_batches,  sum(tar_train_loss_syn)/tar_train_n_batches,
                    tar_train_acc, tar_train_f1
                )
            )
            print tar_train_cm

            print (
                "soruce valid ({:.3f} sec): n={}, st_loss={:.3f} do_loss={:.3f}, cl_loss={:.3f}, sub_loss={:.3f}, syn_loss= {:.3f}, acc={:.3f}, "
                "f1={:.3f}".format(
                    valid_duration, src_n_valid_examples,
                    sum(src_valid_loss_st)/src_valid_n_batches, sum(src_valid_loss_do)/src_valid_n_batches, sum(src_valid_loss_cl)/src_valid_n_batches, sum(src_valid_loss_sub)/src_valid_n_batches,  sum(src_valid_loss_syn)/src_valid_n_batches,
                    src_valid_acc, src_valid_f1
                )
            )
            print src_valid_cm
            print (
                "target valid ({:.3f} sec): n={}, st_loss={:.3f} do_loss={:.3f}, cl_loss={:.3f}, sub_loss={:.3f}, syn_loss={:3f},  acc={:.3f}, "
                "f1={:.3f}".format(
                    valid_duration, tar_n_valid_examples,
                    sum(tar_valid_loss_st)/tar_valid_n_batches, sum(tar_valid_loss_do)/tar_valid_n_batches, sum(tar_valid_loss_cl)/tar_valid_n_batches, sum(tar_valid_loss_sub)/tar_valid_n_batches, sum(tar_valid_loss_syn)/tar_valid_n_batches,
                    tar_valid_acc, tar_valid_f1
                )
            )
            print tar_valid_cm
            print " "
        else:
            print (
                "epoch {}: "
                "source train ({:.2f} sec): n={}, st_loss={:.3f} do_loss={:.3f} cl_loss={:.3f}, sub_loss={:.3f}, syn_loss={:.3f}, "
                "acc={:.3f}, f1={:.3f} | "
                "source valid ({:.2f} sec): n={}, st_loss={:.3f} do_loss={:.3f} cl_loss={:.3f}, sub_loss={:.3f}, syn_loss={:.3f}, "
                "acc={:.3f}, f1={:.3f} | "
                "target train ({:.2f} sec): n={}, st_loss={:.3f} do_loss={:.3f} cl_loss={:.3f}, sub_loss={:.3f}, syn_loss={:.3f}, "
                "acc={:.3f}, f1={:.3f} | "
                "target valid ({:.2f} sec): n={}, st_loss={:.3f} do_loss={:.3f} cl_loss={:.3f},sub_loss={:.3f}, syn_loss={:.3f}, "
                "acc={:.3f}, f1={:.3f}".format(
                    epoch + 1,
                    train_duration, src_n_train_examples,
                    sum(src_train_loss_st)/src_train_n_batches, sum(src_train_loss_do)/src_train_n_batches, sum(src_train_loss_cl)/src_train_n_batches, sum(src_train_loss_sub)/src_train_n_batches, sum(src_train_loss_syn)/src_train_n_batches, src_train_acc, src_train_f1,
                    valid_duration, src_n_valid_examples,
                    sum(src_valid_loss_st)/src_valid_n_batches, sum(src_valid_loss_do)/src_valid_n_batches, sum(src_valid_loss_cl)/src_valid_n_batches, sum(src_valid_loss_sub)/src_valid_n_batches, sum(src_valid_loss_syn)/src_valid_n_batches, src_valid_acc, src_valid_f1,
                    train_duration, tar_n_train_examples,
                    sum(tar_train_loss_st)/tar_train_n_batches, sum(tar_train_loss_do)/tar_train_n_batches, sum(tar_train_loss_cl)/tar_train_n_batches, sum(tar_train_loss_sub)/tar_train_n_batches, sum(tar_train_loss_syn)/tar_train_n_batches, tar_train_acc, tar_train_f1,
                    valid_duration, tar_n_valid_examples,
                    sum(tar_valid_loss_st)/tar_valid_n_batches, sum(tar_valid_loss_do)/tar_valid_n_batches, sum(tar_valid_loss_cl)/tar_valid_n_batches, sum(tar_valid_loss_sub)/tar_valid_n_batches, sum(tar_valid_loss_syn)/tar_valid_n_batches, tar_valid_acc, tar_valid_f1
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


class DeepFeatureNetTrainer(Trainer):

    def __init__(
            self,
            source_dir,
            target_dir,
            output_dir,
            source_n_folds,
            source_fold_idx,
            target_n_folds,
            target_fold_idx,
            batch_size,
            lr,
            input_dims,
            n_classes,
            target_data = 'sleep-edf-sc',
            interval_plot_filter=50,
            interval_save_model=100,
            interval_print_cm=10,
            adap_epoch=3,
            pam_divide=1,
            alpha=10.0,
            beta=10.0
    ):
        super(self.__class__, self).__init__(
            interval_plot_filter=interval_plot_filter,
            interval_save_model=interval_save_model,
            interval_print_cm=interval_print_cm
        )

        self.source_dir = source_dir
        self.source_n_folds = source_n_folds
        self.source_fold_idx = source_fold_idx

        self.target_dir = target_dir
        self.target_n_folds = target_n_folds
        self.target_fold_idx = target_fold_idx
        self.target_data =target_data
        
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = lr
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.adap_epoch =adap_epoch
        self.pam_divide = pam_divide
        self.alpha = alpha
        self.beta = beta

    def _run_epoch(self, sess, network, inputs, targets, subjects, inputs_unseen, targets_unseen, train_op, alpha, beta, gamma, is_train, m=1):
        start_time = time.time()
        y = []
        y_true = []
        src_total_loss_st, src_total_loss_do, src_total_loss_cl, src_total_loss_sub,  src_total_loss_syn, source_n_batches= [], [], [], [], [] ,0
        # for unseen signal
        y_unseen = []
        y_unseen_true = []     
        tar_total_loss_st, tar_total_loss_do, tar_total_loss_cl, tar_total_loss_sub, tar_total_loss_syn, target_n_batches =  [], [], [], [], [] ,0
        
        if is_train:
            print alpha, beta, gamma
            is_shuffle=True

            # downsample source domain into target domain
            class_tar = int(len(targets_unseen) / 5 )
            inputs, targets, subjects = get_balance_class_subject_by_target(inputs, targets, subjects, class_tar)
            # sample source train
            for (x_batch, y_batch, s_batch), (x_batch_unseen, y_batch_unseen) in zip(iterate_minibatches_subject(inputs, targets,subjects, self.batch_size, shuffle=is_shuffle), iterate_minibatches(inputs_unseen, targets_unseen, self.batch_size, shuffle=is_shuffle)):

                feed_dict = {
                    network.input_var: x_batch,
                    network.is_source: True,
                    network.target_var: y_batch,
                    network.subject_var: s_batch,
                    network.alpha: alpha,
                    network.beta: beta,
                    #network.gamma: gamma,
                    network.domain_var: np.ones(self.batch_size , dtype=np.int32)
                }

                _, _, _, _, _, src_loss_st, src_loss_do, src_loss_cl, src_loss_sub, src_loss_syn, y_pred = sess.run(
                    [train_op[0], train_op[1], train_op[2], train_op[3], train_op[4], network.staging_loss,
                     network.domain_loss,
                     network.class_distributor_loss, network.subject_discriminator_loss, network.syn_loss,
                     network.pred_op],
                    feed_dict=feed_dict
                )

                src_total_loss_st.append(src_loss_st)
                src_total_loss_do.append(src_loss_do)
                src_total_loss_cl.append(src_loss_cl)
                src_total_loss_sub.append(src_loss_sub)
                src_total_loss_syn.append(src_loss_syn)

                source_n_batches += 1
                y.append(y_pred)
                y_true.append(y_batch)
                feed_dict = {
                    network.input_var: x_batch_unseen,
                    network.is_source: False,
                    network.alpha: alpha,
                    network.beta: beta,
                    #network.gamma: gamma,
                    network.target_var: y_batch_unseen,
                    network.subject_var: np.zeros(self.batch_size, dtype=np.int32),
                    network.domain_var: np.zeros(self.batch_size, dtype=np.int32)
                }

                _, _, _, _, _,  loss_st, loss_do, loss_cl,loss_sub, loss_syn, y_pred_unseen = sess.run(
                    [train_op[0], train_op[1], train_op[2], train_op[3], train_op[4], network.staging_loss,
                     network.domain_loss,
                     network.class_distributor_loss, network.subject_discriminator_loss, network.syn_loss,
                     network.pred_op],
                    feed_dict=feed_dict
                )
                tar_total_loss_st.append(loss_st)
                tar_total_loss_do.append(loss_do)
                tar_total_loss_cl.append(loss_cl)
                tar_total_loss_sub.append(loss_sub)
                tar_total_loss_syn.append(loss_syn)
                target_n_batches += 1
                y_unseen.append(y_pred_unseen)
                y_unseen_true.append(y_batch_unseen)
        else:
            for (x_batch, y_batch, z_batch) in iterate_minibatches_subject(inputs=inputs, targets=targets, subjects=subjects, batch_size=self.batch_size):
                feed_dict = {
                    network.input_var: x_batch,
                    network.is_source: True,
                    network.target_var: y_batch,
                    network.subject_var: z_batch,
                    network.alpha: 1.0,
                    network.beta: 1.0,
                    #network.gamma: 1.0,
                    network.domain_var: np.ones(self.batch_size , dtype=np.int32)
                }
                _, _, _, _, _, src_loss_st, src_loss_do, src_loss_cl, src_loss_sub, src_loss_syn, y_pred = sess.run(
                    [train_op[0], train_op[1], train_op[2], train_op[3], train_op[4], network.staging_loss,
                     network.domain_loss, network.class_distributor_loss, network.subject_discriminator_loss,
                     network.syn_loss, network.pred_op],
                    feed_dict=feed_dict
                )

                src_total_loss_st.append(src_loss_st)
                src_total_loss_do.append(src_loss_do)
                src_total_loss_cl.append(src_loss_cl)
                src_total_loss_sub.append(src_loss_sub)
                src_total_loss_syn.append(src_loss_syn)
                source_n_batches += 1
                y.append(y_pred)
                y_true.append(y_batch)
                
            for x_batch_unseen, y_batch_unseen in iterate_minibatches(inputs=inputs_unseen, targets=targets_unseen, batch_size=self.batch_size):                
                feed_dict = {
                    network.input_var: x_batch_unseen,
                    network.is_source: False,
                    network.alpha: 1.0,
                    network.beta: 1.0,
                    #network.gamma:1.0,
                    network.target_var: y_batch_unseen,
                    network.domain_var: np.zeros(self.batch_size , dtype=np.int32),
                    network.subject_var: np.zeros(self.batch_size , dtype=np.int32)
                }
                _, _, _, _, _, loss_st, loss_do, loss_cl, loss_sub,  loss_syn, y_pred_unseen = sess.run(
                    [train_op[0], train_op[1], train_op[2], train_op[3], train_op[4], network.staging_loss,
                     network.domain_loss, network.class_distributor_loss,network.subject_discriminator_loss,
                     network.syn_loss, network.pred_op],
                    feed_dict=feed_dict
                )
                tar_total_loss_st.append(loss_st)
                tar_total_loss_do.append(loss_do)
                tar_total_loss_cl.append(loss_cl)
                tar_total_loss_sub.append(loss_sub)
                tar_total_loss_syn.append(loss_syn)
                target_n_batches += 1
                y_unseen.append(y_pred_unseen)
                y_unseen_true.append(y_batch_unseen)
                    
        duration = time.time() - start_time
        '''
        src_total_loss_st /= source_n_batches
        src_total_loss_do /= source_n_batches
        src_total_loss_cl /= source_n_batches
        src_total_loss_syn /= source_n_batches

        tar_total_loss_st /= target_n_batches
        tar_total_loss_do /= target_n_batches
        tar_total_loss_cl /= target_n_batches
        tar_total_loss_syn /= target_n_batches
        
        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)
        total_y_pred_unseen = np.hstack(y_unseen)
        total_y_true_unseen = np.hstack(y_unseen_true)
        '''
        alpha = (sum(src_total_loss_st)/source_n_batches) / (sum(tar_total_loss_st)/target_n_batches)
        # beta = (src_total_loss_do + src_total_loss_cl +src_total_loss_sub) / src_total_loss_syn
        beta = np.mean([sum(src_total_loss_do)/source_n_batches, sum(src_total_loss_cl)/source_n_batches, sum(src_total_loss_sub)/source_n_batches]) / (sum(src_total_loss_syn)/source_n_batches)
        gamma = (sum(src_total_loss_st)/source_n_batches) / ((sum(src_total_loss_do) + sum(src_total_loss_cl)+sum(src_total_loss_sub))/source_n_batches)
        return y_true, y, y_unseen_true, y_unseen, [src_total_loss_st, src_total_loss_do, src_total_loss_cl, src_total_loss_sub, src_total_loss_syn], [tar_total_loss_st, tar_total_loss_do, tar_total_loss_cl, tar_total_loss_sub, tar_total_loss_syn], duration, alpha, beta, gamma, source_n_batches, target_n_batches

    def train(self, n_epochs, resume):
        with tf.Graph().as_default(), tf.Session() as sess:
            # Build training and validation networks
            train_net = DeepFeatureNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=True,
                reuse_params=False,
                use_dropout=True,
                args =['class', 'subject', 'adap']
            )
            valid_net = DeepFeatureNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=False,
                reuse_params=True,
                use_dropout=True,
                args=['class','subject', 'adap']
            )
            # Initialize parameters
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
            # vars for staging
            feature_vars = tf.trainable_variables()[:24]
            staging_vars = tf.trainable_variables()[24:28]
            domain_vars =tf.trainable_variables()[28:32]
            class_vars = tf.trainable_variables()[32:52]
            subject_vars = tf.trainable_variables()[52:]
            # Define optimization operations
            train_op1, grads_and_vars_op1 = adam(
                loss=train_net.domain_loss + train_net.syn_loss,
                lr=self.learning_rate,
                train_vars=domain_vars
            )

            train_op2, grads_and_vars_op2 = adam(
                loss=train_net.class_distributor_loss + train_net.syn_loss,
                lr=self.learning_rate,
                train_vars=class_vars
            )

            train_op3, grads_and_vars_op3 = adam(
                loss=train_net.subject_discriminator_loss + train_net.syn_loss,
                lr=self.learning_rate,
                train_vars=subject_vars
            )

            train_op4, grads_and_vars_op4 = adam(
                loss=train_net.staging_loss,
                lr=self.learning_rate,
                train_vars=staging_vars
            )
            
            train_op5, grads_and_vars_op5 = adam(
                loss=train_net.staging_loss + train_net.syn_loss - (
                            train_net.domain_loss +train_net.subject_discriminator_loss+ train_net.class_distributor_loss ),
                lr=self.learning_rate,
                train_vars=feature_vars
            )

            # Make subdirectory for pretraining
            output_dir = os.path.join(self.output_dir, "fold{}".format(self.target_fold_idx), train_net.name)
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
                data_loader = mass_NonSeqDataLoader(
                    data_dir=self.source_dir,
                    n_folds=self.source_n_folds,
                    fold_idx=self.source_fold_idx
                )
                unseen_data_loader = NonSeqDataLoader(
                    data_dir=self.target_dir,
                    n_folds=self.target_n_folds,
                    fold_idx=self.target_fold_idx,
                    data = self.target_data
                )
                x_train, y_train, s_train, x_valid, y_valid, s_valid = data_loader.load_train_data(oversample=False)
                unseen_x_train, unseen_y_train, unseen_x_valid, unseen_y_valid = unseen_data_loader.load_train_data()

                # Performance history
                all_source_train_loss_staging = []
                all_source_train_loss_domain = []
                all_source_train_loss_class = []
                all_source_train_loss_sub = []
                all_source_train_loss_syn = []

                all_target_train_loss_staging = []
                all_target_train_loss_domain = []
                all_target_train_loss_class =[]
                all_target_train_loss_sub =[]
                all_target_train_loss_syn = []

                all_source_train_acc = []
                all_source_train_f1 = []
                all_target_train_acc = []
                all_target_train_f1 = []

                all_source_valid_loss_staging = []
                all_source_valid_loss_domain = []
                all_source_valid_loss_class = []
                all_source_valid_loss_sub =[]
                all_source_valid_loss_syn = []

                all_target_valid_loss_staging = []
                all_target_valid_loss_domain = []
                all_target_valid_loss_class = []
                all_target_valid_loss_sub=[]
                all_target_valid_loss_syn = []

                all_source_valid_acc = []
                all_source_valid_f1 = []
                all_target_valid_acc = []
                all_target_valid_f1 = []
            
            alpha_ori, beta_ori, gamma_ori = 1, 1, 1
            alpha, beta, gamma = 1.0, 1.0, 1.0
            # Loop each epoch
            for epoch in xrange(sess.run(global_step), n_epochs):
                # Update parameters and compute loss of training set
                if epoch < self.adap_epoch:
                    alpha_ori = 1.0
                    beta_ori = 1.0
                    gamma_ori= 1.0
                    #alpha, beta, gamma = 0.0, 0.0, 0.0
                else:
                    if  epoch % self.adap_epoch == 0:
                        alpha_ori = alpha/self.adap_epoch/self.pam_divide
                        beta_ori = beta/self.adap_epoch/self.pam_divide
                        gamma_ori = gamma / self.adap_epoch
                        alpha, beta, gamma = 0.0, 0.0, 0.0

                y_true_train, y_pred_train, y_true_train_unseen, y_pred_train_unseen, src_train_losses, tar_train_losses, train_duration, alpha_temp, beta_temp, gamma_temp, src_train_n_batches, tar_train_n_batches = \
                    self._run_epoch(
                        sess=sess, network=train_net,
                        inputs=x_train, targets=y_train, subjects =s_train,
                        inputs_unseen=unseen_x_train, targets_unseen=unseen_y_train,
                        train_op=[train_op1, train_op2,train_op3,  train_op4, train_op5],
                        alpha=alpha_ori, beta=beta_ori, gamma=gamma_ori,
                        is_train=True
                    )
                
                alpha += alpha_temp
                beta += beta_temp
                gamma += gamma_temp

                src_n_train_examples = len(y_true_train)
                for idx in range(src_n_train_examples):
                    y_true = y_true_train[idx]
                    y_pred = y_pred_train[idx]
                    src_train_cm = confusion_matrix(y_true, y_pred)
                    src_train_acc = np.mean(y_true == y_pred)
                    src_train_f1 = f1_score(y_true, y_pred, average="macro")
                    all_source_train_acc.append(src_train_acc)
                    all_source_train_f1.append(src_train_f1)

                tar_n_train_examples = len(y_true_train_unseen)
                for idx in range(tar_n_train_examples):
                    y_true = y_true_train_unseen[idx]
                    y_pred = y_pred_train_unseen[idx]
                    tar_train_cm = confusion_matrix(y_true, y_pred)
                    tar_train_acc = np.mean(y_true == y_pred)
                    tar_train_f1 = f1_score(y_true, y_pred, average="macro")
                    all_target_train_acc.append(tar_train_acc)
                    all_target_train_f1.append(tar_train_f1)

                # Evaluate the model on the validation set
                y_true_val, y_pred_val, y_true_val_unseen, y_pred_val_unseen, src_valid_losses, tar_valid_losses, valid_duration, _, _, _, src_valid_n_batches, tar_valid_n_batches= \
                    self._run_epoch(
                        sess=sess, network=valid_net,
                        inputs=x_valid, targets=y_valid, subjects=s_valid,
                        inputs_unseen=unseen_x_valid, targets_unseen=unseen_y_valid,
                        train_op=[tf.no_op(), tf.no_op(), tf.no_op(), tf.no_op(), tf.no_op()], alpha=1.0, beta=1.0, gamma=1.0,
                        is_train=False
                    )
                
                src_n_valid_examples = len(y_true_val)
                for idx in range(src_n_valid_examples):
                    y_true = y_true_val[idx]
                    y_pred = y_pred_val[idx]
                    src_valid_cm = confusion_matrix(y_true, y_pred)
                    src_valid_acc = np.mean(y_true == y_pred)
                    src_valid_f1 = f1_score(y_true, y_pred, average="macro")
                    all_source_valid_acc.append(src_valid_acc )
                    all_source_valid_f1.append(src_valid_f1)

                tar_n_valid_examples = len(y_true_val_unseen)
                for idx in range(tar_n_valid_examples):
                    y_true = y_true_val_unseen[idx]
                    y_pred = y_pred_val_unseen[idx]
                    tar_valid_cm = confusion_matrix(y_true, y_pred)
                    tar_valid_acc = np.mean(y_true == y_pred)
                    tar_valid_f1 = f1_score(y_true, y_pred, average="macro")
                    all_target_valid_acc.append(tar_valid_acc )
                    all_target_valid_f1.append(tar_valid_f1)

                all_source_train_loss_staging += src_train_losses[0]
                all_source_train_loss_domain += src_train_losses[1]
                all_source_train_loss_class += src_train_losses[2]
                all_source_train_loss_sub +=src_train_losses[3]
                all_source_train_loss_syn += src_train_losses[4]

                all_target_train_loss_staging += tar_train_losses[0]
                all_target_train_loss_domain += tar_train_losses[1]
                all_target_train_loss_class += tar_train_losses[2]
                all_target_train_loss_sub += tar_train_losses[3]
                all_target_train_loss_syn += tar_train_losses[4]

                all_source_valid_loss_staging += src_valid_losses[0]
                all_source_valid_loss_domain += src_valid_losses[1]
                all_source_valid_loss_class += src_valid_losses[2]
                all_source_valid_loss_sub += src_valid_losses[3]
                all_source_valid_loss_syn += src_valid_losses[4]

                all_target_valid_loss_staging += tar_valid_losses[0]
                all_target_valid_loss_domain += tar_valid_losses[1]
                all_target_valid_loss_class += tar_valid_losses[2]
                all_target_valid_loss_sub += tar_valid_losses[3]
                all_target_valid_loss_syn += tar_valid_losses[4]

                # Report performance
                self.print_performance(
                    sess, train_net.name,
                    src_n_train_examples, src_n_valid_examples, tar_n_train_examples, tar_n_valid_examples,
                    src_train_cm, src_valid_cm, tar_train_cm, tar_valid_cm, epoch, n_epochs,
                    train_duration, src_train_losses[0], src_train_losses[1], src_train_losses[2], src_train_losses[3], src_train_losses[4],src_train_acc, src_train_f1,
                    tar_train_losses[0], tar_train_losses[1], tar_train_losses[2], tar_train_losses[3],  tar_train_losses[4], tar_train_acc, tar_train_f1,
                    valid_duration, src_valid_losses[0], src_valid_losses[1], src_valid_losses[2],src_valid_losses[3], src_valid_losses[4],src_valid_acc, src_valid_f1,
                    tar_valid_losses[0], tar_valid_losses[1], tar_valid_losses[2],tar_valid_losses[3],tar_valid_losses[4],  tar_valid_acc, tar_valid_f1,
                    src_train_n_batches, tar_train_n_batches, src_valid_n_batches, tar_valid_n_batches
                )  
                
                #valid_loss_de = all_valid_loss_de,train_loss_de=all_train_loss_de,    
                # Save performance history
                np.savez(
                    os.path.join(output_dir, "perf_fold{}.npz".format(self.target_fold_idx)),
                    src_train_loss_st = all_source_train_loss_staging, src_train_loss_do = all_source_train_loss_domain,
                    src_train_loss_cl = all_source_train_loss_class,  src_train_loss_sub = all_source_train_loss_sub,
                    src_train_loss_syn = all_source_train_loss_syn,
                    tar_train_loss_st=all_target_train_loss_staging, tar_train_loss_do=all_target_train_loss_domain,
                    tar_train_loss_cl=all_target_train_loss_class, tar_train_loss_sub = all_target_train_loss_sub,
                    tar_train_loss_syn=all_target_train_loss_syn,
                    src_valid_loss_st=all_source_valid_loss_staging, src_valid_loss_do=all_source_valid_loss_domain,
                    src_valid_loss_cl=all_source_valid_loss_class,  src_valid_loss_sub = all_source_valid_loss_sub,
                    src_valid_loss_syn=all_source_valid_loss_syn,
                    tar_valid_loss_st=all_target_valid_loss_staging, tar_valid_loss_do=all_target_valid_loss_domain,
                    tar_valid_loss_cl=all_target_valid_loss_class, tar_valid_loss_sub = all_target_valid_loss_sub,
                    tar_valid_loss_syn=all_target_valid_loss_syn,
                    src_train_acc=all_source_train_acc, src_valid_acc=all_source_valid_acc,
                    tar_train_acc=all_target_train_acc, tar_valid_acc=all_target_valid_acc,
                    src_train_f1=all_source_train_f1, src_valid_f1=all_source_valid_f1,
                    tar_train_f1=all_target_train_f1, tar_valid_f1=all_target_valid_f1,
                    y_true_val=np.asarray(y_true_val), y_pred_val=np.asarray(y_pred_val),
                    y_true_val_unseen = np.asarray(y_true_val_unseen), y_pred_val_unseen = np.asarray(y_pred_val_unseen)

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
                        output_dir, "model_fold{}.ckpt".format(self.target_fold_idx)
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
                            "params_fold{}.npz".format(self.target_fold_idx)),
                        **save_dict
                    )
                    duration = time.time() - start_time
                    print "Saved trained parameters ({:.3f} sec)".format(duration)

        print "Finish pre-training"
        return os.path.join(output_dir, "params_fold{}.npz".format(self.target_fold_idx))

class DeepSleepNetTrainer(Trainer):
    def __init__(
            self,
            source_dir,
            target_dir,
            output_dir,
            source_n_folds,
            source_fold_idx,
            target_n_folds,
            target_fold_idx,
            batch_size,
            lr,
            input_dims,
            n_classes,
            seq_length,
            n_rnn_layers,
            return_last,
            target_data = 'sleep-edf-sc',
            interval_plot_filter=50,
            interval_save_model=100,
            interval_print_cm=10,
            adap_epoch =3, 
            pam_divide=1,
            alpha=10.0,
            beta=10.0
    ):
        super(self.__class__, self).__init__(
            interval_plot_filter=interval_plot_filter,
            interval_save_model=interval_save_model,
            interval_print_cm=interval_print_cm
        )

        self.source_dir = source_dir
        self.source_n_folds = source_n_folds
        self.source_fold_idx = source_fold_idx

        self.target_dir = target_dir
        self.target_n_folds = target_n_folds
        self.target_fold_idx = target_fold_idx
        self.target_data = target_data

        self.output_dir = output_dir

        self.batch_size = batch_size
        self.learning_rate = lr
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.return_last = return_last
        self.adap_epoch =adap_epoch
        self.pam_divide = pam_divide
        self.alpha = alpha
        self.beta = beta

    def _run_epoch(self, sess, network, inputs, targets, subjects, inputs_unseen, targets_unseen, train_op, alpha, beta, is_train, m=1):
        start_time = time.time()
        y = []
        y_true = []
        src_total_loss_st, src_total_loss_do, src_total_loss_cl, src_total_loss_sub,  src_total_loss_syn, source_n_batches= [], [], [], [] ,[], 0
        # for unseen signal
        y_unseen = []
        y_unseen_true = []                
        tar_total_loss_st, tar_total_loss_do, tar_total_loss_cl, tar_total_loss_sub,  tar_total_loss_syn, target_n_batches = [], [], [], [] ,[], 0
        # for unseen signal
        
        if is_train:
            print alpha, beta
            inputs, targets, subjects = shuffle(inputs, targets, subjects)
            inputs_unseen, targets_unseen = shuffle(inputs_unseen, targets_unseen)
            for (sub_idx, each_data), (sub_idx_unseen, each_data_unseen) in zip(enumerate(itertools.izip(inputs, targets, subjects)), enumerate(itertools.izip(inputs_unseen, targets_unseen))):
                each_x, each_y, each_s = each_data
                each_x_unseen, each_y_unseen = each_data_unseen

                # Initialize state of LSTM - Bidirectional LSTM
                fw_state = sess.run(network.fw_initial_state)
                bw_state = sess.run(network.bw_initial_state)

                for (x_batch, y_batch, s_batch), (x_batch_unseen, y_batch_unseen) in zip(iterate_batch_seq_minibatches_subject(inputs=each_x, targets=each_y,subjects=each_s,
                                                                        batch_size=self.batch_size, seq_length=self.seq_length),
                                                                        iterate_batch_seq_minibatches(inputs=each_x_unseen, targets=each_y_unseen,
                                                                                batch_size=self.batch_size, seq_length=self.seq_length)):
                    feed_dict = {
                        network.input_var: x_batch,
                        network.is_source: True,
                        network.target_var: y_batch,
                        network.subject_var: s_batch,
                        network.alpha: alpha,
                        network.beta:  beta,
                        network.domain_var: np.ones(self.batch_size * self.seq_length, dtype=np.int32)
                    }

                    for i, (c, h) in enumerate(network.fw_initial_state):
                        feed_dict[c] = fw_state[i].c
                        feed_dict[h] = fw_state[i].h

                    for i, (c, h) in enumerate(network.bw_initial_state):
                        feed_dict[c] = bw_state[i].c
                        feed_dict[h] = bw_state[i].h

                    _, _, _, _,_, src_loss_st, src_loss_do, src_loss_cl, src_loss_sub,  src_loss_syn, y_pred = sess.run(
                        [train_op[0], train_op[1], train_op[2], train_op[3], train_op[4],  network.staging_loss, network.domain_loss,
                         network.class_distributor_loss, network.subject_discriminator_loss,  network.syn_loss, network.pred_op],
                        feed_dict=feed_dict
                    )

                    src_total_loss_st.append( src_loss_st)
                    src_total_loss_do.append( src_loss_do)
                    src_total_loss_cl.append( src_loss_cl)
                    src_total_loss_sub.append(src_loss_sub)
                    src_total_loss_syn.append( src_loss_syn)
                    
                    source_n_batches += 1
                    y.append(y_pred)
                    y_true.append(y_batch)

                    feed_dict = {
                        network.input_var: x_batch_unseen,
                        network.is_source: False,
                        network.alpha: alpha,
                        network.beta: beta,
                        #network.gamma: gamma,
                        network.target_var: y_batch_unseen,
                        network.subject_var: np.zeros(self.batch_size * self.seq_length, dtype=np.int32),
                        network.domain_var: np.zeros(self.batch_size * self.seq_length, dtype=np.int32)
                    }

                    for i, (c, h) in enumerate(network.fw_initial_state):
                        feed_dict[c] = fw_state[i].c
                        feed_dict[h] = fw_state[i].h

                    for i, (c, h) in enumerate(network.bw_initial_state):
                        feed_dict[c] = bw_state[i].c
                        feed_dict[h] = bw_state[i].h

                    _, _, _, _, _, loss_st, loss_do, loss_cl, loss_sub, loss_syn, y_pred_unseen = sess.run(
                        [train_op[0], train_op[1], train_op[2], train_op[3], train_op[4], network.staging_loss, network.domain_loss,
                         network.class_distributor_loss, network.subject_discriminator_loss, network.syn_loss, network.pred_op],
                        feed_dict=feed_dict
                    )
                    tar_total_loss_st.append( loss_st)
                    tar_total_loss_do.append( loss_do)
                    tar_total_loss_cl.append(  loss_cl)
                    tar_total_loss_sub.append(loss_sub)
                    tar_total_loss_syn.append(loss_syn)
                    target_n_batches += 1
                    y_unseen.append(y_pred_unseen)
                    y_unseen_true.append(y_batch_unseen)
        else:
                
            for sub_idx, each_data in enumerate(itertools.izip(inputs, targets, subjects)):
    
                each_x, each_y,each_s = each_data
    
                # Initialize state of LSTM - Bidirectional LSTM
                fw_state = sess.run(network.fw_initial_state)
                bw_state = sess.run(network.bw_initial_state)
    
                for (x_batch, y_batch,z_batch) in iterate_batch_seq_minibatches_subject(inputs=each_x, targets=each_y, subjects=each_s, batch_size=self.batch_size, seq_length=self.seq_length):
    
                    feed_dict = {
                        network.input_var: x_batch,
                        network.is_source: True,
                        network.target_var: y_batch,
                        network.subject_var: z_batch,
                        network.alpha: 1.0,
                        network.beta:1.0,
                        #network.gamma: 1.0,
                        network.domain_var: np.ones(self.batch_size*self.seq_length, dtype=np.int32)
                    }
    
                    for i, (c, h) in enumerate(network.fw_initial_state):
                        feed_dict[c] = fw_state[i].c
                        feed_dict[h] = fw_state[i].h
    
                    for i, (c, h) in enumerate(network.bw_initial_state):
                        feed_dict[c] = bw_state[i].c
                        feed_dict[h] = bw_state[i].h
    
                    _,_,_,_,_,  src_loss_st, src_loss_do, src_loss_cl,src_loss_sub, src_loss_syn,  y_pred= sess.run(
                        [train_op[0], train_op[1], train_op[2], train_op[3], train_op[4],
                         network.staging_loss, network.domain_loss, network.class_distributor_loss,
                         network.subject_discriminator_loss, network.syn_loss, network.pred_op],
                        feed_dict=feed_dict
                    )
    
                    src_total_loss_st.append( src_loss_st)
                    src_total_loss_do.append(  src_loss_do)
                    src_total_loss_cl.append( src_loss_cl)
                    src_total_loss_sub.append(src_loss_sub)
                    src_total_loss_syn.append( src_loss_syn)
                    source_n_batches += 1
                    y.append(y_pred)
                    y_true.append(y_batch)
    
            for sub_idx_unseen, each_data_unseen in enumerate(itertools.izip(inputs_unseen, targets_unseen)):
    
                each_x_unseen, each_y_unseen = each_data_unseen
    
                # Initialize state of LSTM - Bidirectional LSTM
                fw_state = sess.run(network.fw_initial_state)
                bw_state = sess.run(network.bw_initial_state)
                for x_batch_unseen, y_batch_unseen in iterate_batch_seq_minibatches(inputs=each_x_unseen, targets=each_y_unseen,
                                                      batch_size=self.batch_size, seq_length=self.seq_length):
                    feed_dict = {
                        network.input_var: x_batch_unseen,
                        network.is_source: False,
                        network.alpha: 1.0,
                        network.beta: 1.0,
                        #network.gamma: 1.0,
                        network.target_var: y_batch_unseen,
                        network.domain_var: np.zeros(self.batch_size * self.seq_length, dtype=np.int32),
                        network.subject_var: np.zeros(self.batch_size * self.seq_length, dtype=np.int32)
                    }
    
                    for i, (c, h) in enumerate(network.fw_initial_state):
                        feed_dict[c] = fw_state[i].c
                        feed_dict[h] = fw_state[i].h
    
                    for i, (c, h) in enumerate(network.bw_initial_state):
                        feed_dict[c] = bw_state[i].c
                        feed_dict[h] = bw_state[i].h
    
                    _,_,_,_,_, loss_st, loss_do, loss_cl, loss_sub, loss_syn,  y_pred_unseen = sess.run(
                        [train_op[0], train_op[1], train_op[2], train_op[3], train_op[4],
                         network.staging_loss, network.domain_loss, network.class_distributor_loss,
                         network.subject_discriminator_loss, network.syn_loss, network.pred_op],
                        feed_dict=feed_dict
                    )
                    tar_total_loss_st.append( loss_st)
                    tar_total_loss_do.append( loss_do)
                    tar_total_loss_cl.append( loss_cl)
                    tar_total_loss_sub.append(loss_sub)
                    tar_total_loss_syn.append( loss_syn)
                    target_n_batches += 1
                    y_unseen.append(y_pred_unseen)
                    y_unseen_true.append(y_batch_unseen)
            
        duration = time.time() - start_time
        '''
        src_total_loss_st /= source_n_batches
        src_total_loss_do /= source_n_batches
        src_total_loss_cl /= source_n_batches
        src_total_loss_syn /=source_n_batches

        tar_total_loss_st /= target_n_batches
        tar_total_loss_do /= target_n_batches
        tar_total_loss_cl /= target_n_batches
        tar_total_loss_syn /= target_n_batches
        
        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)
        total_y_pred_unseen = np.hstack(y_unseen)
        total_y_true_unseen = np.hstack(y_unseen_true)
        '''
        alpha = (sum(src_total_loss_st)/source_n_batches)/ (sum(tar_total_loss_st)/target_n_batches)
        #beta = (src_total_loss_do + src_total_loss_cl +src_total_loss_sub) / src_total_loss_syn
        beta = (np.mean([sum(src_total_loss_do) , sum(src_total_loss_cl), sum(src_total_loss_sub)] )/source_n_batches) /(sum(src_total_loss_syn)/source_n_batches)
       
        return y_true, y, y_unseen_true, y_unseen, [src_total_loss_st, src_total_loss_do, src_total_loss_cl, src_total_loss_sub,  src_total_loss_syn], [tar_total_loss_st, tar_total_loss_do, tar_total_loss_cl, tar_total_loss_sub, tar_total_loss_syn], duration, alpha, beta,source_n_batches, target_n_batches

    def finetune(self, pretrained_model_path, n_epochs, resume):
        pretrained_model_name = "deepfeaturenet"

        with tf.Graph().as_default(), tf.Session() as sess:
            # Build training and validation networks
            train_net = DeepSleepNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                seq_length=self.seq_length,
                n_rnn_layers=self.n_rnn_layers,
                return_last=self.return_last,
                is_train=True,
                reuse_params=False,
                use_dropout_feature=True,
                use_dropout_sequence=True,
                alpha = self.alpha,
                args = ['class','subject','adap']
            )
            valid_net = DeepSleepNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                seq_length=self.seq_length,
                n_rnn_layers=self.n_rnn_layers,
                return_last=self.return_last,
                is_train=False,
                reuse_params=True,
                use_dropout_feature=True,
                use_dropout_sequence=True,
                alpha=self.alpha,
                args=['class','subject', 'adap']
            )

            # Initialize parameters
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

            feature_vars = tf.trainable_variables()[:47]
            staging_vars = tf.trainable_variables()[47:51]
            domain_vars = tf.trainable_variables()[51:55]
            class_vars = tf.trainable_variables()[55:75]
            subject_vars = tf.trainable_variables()[75:]
            # Define optimization operations
            train_op1, grads_and_vars_op1 = adam(
                loss= train_net.domain_loss + train_net.syn_loss,
                lr=self.learning_rate,
                train_vars=domain_vars
            )
            
            train_op2, grads_and_vars_op2 = adam(
                loss= train_net.class_distributor_loss + train_net.syn_loss,
                lr=self.learning_rate,
                train_vars = class_vars
            )

            train_op3, grads_and_vars_op3 = adam(
                loss= train_net.subject_discriminator_loss + train_net.syn_loss,
                lr=self.learning_rate,
                train_vars = subject_vars
            )

            train_op4, grads_and_vars_op4 = adam(
                loss=train_net.staging_loss,
                lr=self.learning_rate,
                train_vars=staging_vars
            )
            train_op5, grads_and_vars_op5 = adam(
                loss=train_net.staging_loss + train_net.syn_loss - (train_net.domain_loss + train_net.subject_discriminator_loss+train_net.class_distributor_loss ),
                lr= self.learning_rate,
                train_vars=feature_vars
            )
            
            # Make subdirectory for pretraining
            output_dir = os.path.join(self.output_dir, "fold{}".format(self.target_fold_idx), train_net.name)
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

            # Resume the training if applicable
            load_pretrain = False
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
                print " "
            print "[{}] Start training ...\n".format(datetime.now())
            # Load data
            if sess.run(global_step) < n_epochs:
                data_loader = mass_SeqDataLoader(
                    data_dir=self.source_dir,
                    n_folds=self.source_n_folds,
                    fold_idx=self.source_fold_idx
                )
                unseen_data_loader = SeqDataLoader(
                    data_dir=self.target_dir,
                    n_folds=self.target_n_folds,
                    fold_idx=self.target_fold_idx,
                    data = self.target_data
                )

                x_train, y_train, s_train, x_valid, y_valid, s_valid = data_loader.load_train_data()
                unseen_x_train, unseen_y_train, unseen_x_valid, unseen_y_valid = unseen_data_loader.load_train_data()

                # Performance history
                all_source_train_loss_staging = []
                all_source_train_loss_domain = []
                all_source_train_loss_class = []
                all_source_train_loss_sub = []
                all_source_train_loss_syn =[]
                
                all_target_train_loss_staging = []
                all_target_train_loss_domain = []
                all_target_train_loss_class = []
                all_target_train_loss_sub = []
                all_target_train_loss_syn = []

                all_source_train_acc = []
                all_source_train_f1 = []
                all_target_train_acc = []
                all_target_train_f1 = []

                all_source_valid_loss_staging = []
                all_source_valid_loss_domain = []
                all_source_valid_loss_class = []
                all_source_valid_loss_sub = []
                all_source_valid_loss_syn = []

                all_target_valid_loss_staging = []
                all_target_valid_loss_domain = []
                all_target_valid_loss_class = []
                all_target_valid_loss_sub =[]
                all_target_valid_loss_syn = []

                all_source_valid_acc = []
                all_source_valid_f1 =[]
                all_target_valid_acc = []
                all_target_valid_f1 = []

            alpha_ori, beta_ori =1.0, 1.0
            alpha, beta = 1.0 , 1.0
            # Loop each epoch
            for epoch in xrange(sess.run(global_step), n_epochs):
                # Update parameters and compute loss of training set
                if epoch < self.adap_epoch:
                    alpha, beta = 1.0, 1.0

                if epoch % self.adap_epoch == 0 and epoch !=0 :
                    alpha_ori = alpha/self.adap_epoch/self.pam_divide
                    beta_ori = beta/self.adap_epoch/self.pam_divide
                    alpha, beta = 0.0, 0.0
                    
                y_true_train, y_pred_train, y_true_train_unseen, y_pred_train_unseen, src_train_losses, tar_train_losses, train_duration, alpha_temp, beta_temp, src_train_n_batches, tar_train_n_batches = \
                    self._run_epoch(
                        sess=sess, network=train_net,
                        inputs=x_train, targets=y_train, subjects =s_train,
                        inputs_unseen=unseen_x_train, targets_unseen=unseen_y_train,
                        train_op=[train_op1, train_op2, train_op3, train_op4, train_op5],
                        alpha=alpha_ori, beta=beta_ori,
                        is_train=True
                    )
                
                alpha += alpha_temp
                beta += beta_temp
                
                src_n_train_examples = len(y_true_train)
                for idx in range(src_n_train_examples):
                    y_true = y_true_train[idx]
                    y_pred = y_pred_train[idx]
                    src_train_cm = confusion_matrix(y_true, y_pred)
                    src_train_acc = np.mean(y_true == y_pred)
                    src_train_f1 = f1_score(y_true, y_pred, average="macro")
                    all_source_train_acc.append(src_train_acc)
                    all_source_train_f1.append(src_train_f1)

                tar_n_train_examples = len(y_true_train_unseen)
                for idx in range(tar_n_train_examples):
                    y_true = y_true_train_unseen[idx]
                    y_pred = y_pred_train_unseen[idx]
                    tar_train_cm = confusion_matrix(y_true, y_pred)
                    tar_train_acc = np.mean(y_true == y_pred)
                    tar_train_f1 = f1_score(y_true, y_pred, average="macro")
                    all_target_train_acc.append(tar_train_acc)
                    all_target_train_f1.append(tar_train_f1)

                # Evaluate the model on the validation set
                y_true_val, y_pred_val, y_true_val_unseen, y_pred_val_unseen, src_valid_losses, tar_valid_losses, valid_duration, _, _, src_valid_n_batches, tar_valid_n_batches = \
                    self._run_epoch(
                        sess=sess, network=valid_net,
                        inputs=x_valid, targets=y_valid, subjects=s_valid,
                        inputs_unseen=unseen_x_valid, targets_unseen=unseen_y_valid,
                        train_op=[tf.no_op(), tf.no_op(),tf.no_op(), tf.no_op(), tf.no_op()], alpha=1.0, beta=1.0,
                        is_train=False
                    )

                src_n_valid_examples = len(y_true_val)
                for idx in range(src_n_valid_examples):
                    y_true = y_true_val[idx]
                    y_pred = y_pred_val[idx]
                    src_valid_cm = confusion_matrix(y_true, y_pred)
                    src_valid_acc = np.mean(y_true == y_pred)
                    src_valid_f1 = f1_score(y_true, y_pred, average="macro")
                    all_source_valid_acc.append(src_valid_acc)
                    all_source_valid_f1.append(src_valid_f1)

                tar_n_valid_examples = len(y_true_val_unseen)
                for idx in range(tar_n_valid_examples):
                    y_true = y_true_val_unseen[idx]
                    y_pred = y_pred_val_unseen[idx]
                    tar_valid_cm = confusion_matrix(y_true, y_pred)
                    tar_valid_acc = np.mean(y_true == y_pred)
                    tar_valid_f1 = f1_score(y_true, y_pred, average="macro")
                    all_target_valid_acc.append(tar_valid_acc)
                    all_target_valid_f1.append(tar_valid_f1)

                all_source_train_loss_staging += src_train_losses[0]
                all_source_train_loss_domain += src_train_losses[1]
                all_source_train_loss_class += src_train_losses[2]
                all_source_train_loss_sub += src_train_losses[3]
                all_source_train_loss_syn += src_train_losses[4]

                all_target_train_loss_staging += tar_train_losses[0]
                all_target_train_loss_domain += tar_train_losses[1]
                all_target_train_loss_class += tar_train_losses[2]
                all_target_train_loss_sub += tar_train_losses[3]
                all_target_train_loss_syn += tar_train_losses[4]

                all_source_valid_loss_staging += src_valid_losses[0]
                all_source_valid_loss_domain += src_valid_losses[1]
                all_source_valid_loss_class += src_valid_losses[2]
                all_source_valid_loss_sub += src_valid_losses[3]
                all_source_valid_loss_syn += src_valid_losses[4]

                all_target_valid_loss_staging += tar_valid_losses[0]
                all_target_valid_loss_domain += tar_valid_losses[1]
                all_target_valid_loss_class += tar_valid_losses[2]
                all_target_valid_loss_sub += tar_valid_losses[3]
                all_target_valid_loss_syn += tar_valid_losses[4]

                self.print_performance(
                    sess, train_net.name,
                    src_n_train_examples, src_n_valid_examples, tar_n_train_examples, tar_n_valid_examples,
                    src_train_cm, src_valid_cm, tar_train_cm, tar_valid_cm, epoch, n_epochs,
                    train_duration, src_train_losses[0], src_train_losses[1], src_train_losses[2], src_train_losses[3], src_train_losses[4], src_train_acc, src_train_f1,
                    tar_train_losses[0], tar_train_losses[1], tar_train_losses[2], tar_train_losses[3], tar_train_losses[4], tar_train_acc, tar_train_f1,
                    valid_duration, src_valid_losses[0], src_valid_losses[1], src_valid_losses[2],src_valid_losses[3], src_valid_losses[4], src_valid_acc, src_valid_f1,
                    tar_valid_losses[0], tar_valid_losses[1], tar_valid_losses[2],tar_valid_losses[3], tar_valid_losses[4], tar_valid_acc, tar_valid_f1,
                    src_train_n_batches, tar_train_n_batches, src_valid_n_batches, tar_valid_n_batches

                )                
                # Save performance history
                np.savez(
                    os.path.join(output_dir, "perf_fold{}.npz".format(self.target_fold_idx)),
                    src_train_loss_st = all_source_train_loss_staging, src_train_loss_do = all_source_train_loss_domain,
                    src_train_loss_cl = all_source_train_loss_class,  src_train_loss_sub = all_source_train_loss_sub,
                    src_train_loss_syn = all_source_train_loss_syn,
                    tar_train_loss_st=all_target_train_loss_staging, tar_train_loss_do=all_target_train_loss_domain,
                    tar_train_loss_cl=all_target_train_loss_class, tar_train_loss_sub = all_target_train_loss_sub,
                    tar_train_loss_syn=all_target_train_loss_syn,
                    src_valid_loss_st=all_source_valid_loss_staging, src_valid_loss_do=all_source_valid_loss_domain,
                    src_valid_loss_cl=all_source_valid_loss_class,  src_valid_loss_sub = all_source_valid_loss_sub,
                    src_valid_loss_syn=all_source_valid_loss_syn,
                    tar_valid_loss_st=all_target_valid_loss_staging, tar_valid_loss_do=all_target_valid_loss_domain,
                    tar_valid_loss_cl=all_target_valid_loss_class, tar_valid_loss_sub= all_target_valid_loss_sub,
                    tar_valid_loss_syn=all_target_valid_loss_syn,
                    src_train_acc=all_source_train_acc, src_valid_acc=all_source_valid_acc,
                    tar_train_acc=all_target_train_acc, tar_valid_acc=all_target_valid_acc,
                    src_train_f1=all_source_train_f1, src_valid_f1=all_source_valid_f1,
                    tar_train_f1=all_target_train_f1, tar_valid_f1=all_target_valid_f1,
                    y_true_val=np.asarray(y_true_val), y_pred_val=np.asarray(y_pred_val),
                    y_true_val_unseen = np.asarray(y_true_val_unseen), y_pred_val_unseen = np.asarray(y_pred_val_unseen)

                )

                # Save checkpoint
                sess.run(tf.assign(global_step, epoch + 1))
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_path = os.path.join(
                        output_dir, "model_fold{}.ckpt".format(self.target_fold_idx)
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
                            "params_fold{}.npz".format(self.target_fold_idx)),
                        **save_dict
                    )
                    duration = time.time() - start_time
                    print "Saved trained parameters ({:.3f} sec)".format(duration)