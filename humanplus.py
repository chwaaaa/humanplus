import matplotlib
matplotlib.use('Agg')
import os
#str = "export MPLBACKEND= 'Agg' "
#os.system(str)

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')

from io import BytesIO
from footer import footer
import streamlit as st
import time
import pdb
import numpy as np
import tensorflow as tf
import itertools

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import pandas as pd
import altair as alt

from datetime import time as dtime
import datetime

import tempfile

import torch 
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from deepsleep.model import DeepSleepNet
from deepsleep.nn import *

from deepsleep.utils import iterate_batch_seq_minibatches

import zipfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file, allow_pickle=True) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    data = np.squeeze(data)
    data = data[:, :, np.newaxis, np.newaxis]
    data = data.astype(np.float32)
    labels = labels.astype(np.int32)

    return data, labels, sampling_rate

def plot_confusion_matrix(cm,
                          target_names= np.asarray(['W', 'N1', 'N2', 'N3', 'REM']),
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)





def plot_hypnogram(y_true, y_pred,
                          title=None,c='brown'
                          ):
    if not title:
        title = 'hynpogram'

    classes = np.asarray(['W', 'N1', 'N2', 'N3', 'REM'])

    fig, ax = plt.subplots(2,1)
    c = 'brown'
    x_axis = np.arange(y_pred.shape[0])
    ax[0].plot(x_axis, y_pred, color=c)

    # We want to show all ticks...
    ax[0].set(xticks=np.arange(0, y_true.shape[0], 3600/30),
           yticks=np.arange(classes.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=np.arange(0, y_pred.shape[0], 3600/30)/120, yticklabels=classes,
           xlabel='hour (h)', aspect=30)
    ax[0].set_ylabel('True', rotation=0, fontsize = 12, labelpad=40, color=c)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].yaxis.set_ticks_position('left')
    ax[0].xaxis.set_ticks_position('bottom')

    c = 'blue'

    ax[1].plot(x_axis, y_pred, color=c)

    ax[1].set(xticks=np.arange(0, y_true.shape[0], 3600/30),
           yticks=np.arange(classes.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=np.arange(0, y_pred.shape[0], 3600/30)/120, yticklabels=classes,
           xlabel='hour (h)', aspect=30)
    ax[1].set_ylabel('Predicted', rotation=0, fontsize = 12, labelpad=40, color=c)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].yaxis.set_ticks_position('left')
    ax[1].xaxis.set_ticks_position('bottom')

    fig.tight_layout()
    return fig

def print_performance(sess, network_name, n_examples, duration, loss, cm, acc, f1):
    # Get regularization loss
    reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
    reg_loss_value = sess.run(reg_loss)

    # Print performance
    print(
        "duration={:.3f} sec, n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
        "f1={:.3f}".format(
            duration, n_examples, loss, reg_loss_value, acc, f1
        )
    )
    print(cm)
    print(" ")


def custom_run_epoch(
    sess,
    network,
    inputs,
    targets,
    train_op,
    is_train,
):
    start_time = time.time()
    y = []
    y_true = []
    all_fw_memory_cells = []
    all_bw_memory_cells = []
    total_loss, n_batches = 0.0, 0
    # Initialize state of LSTM - Bidirectional LSTM
    fw_state = sess.run(network.fw_initial_state)
    bw_state = sess.run(network.bw_initial_state)

    # Prepare storage for memory cells
    n_all_data = len(inputs)
    extra = n_all_data % network.seq_length
    n_data = n_all_data - extra
    cell_size = 512
    fw_memory_cells = np.zeros((n_data, network.n_rnn_layers, cell_size))
    bw_memory_cells = np.zeros((n_data, network.n_rnn_layers, cell_size))
    seq_idx = 0

    # Store prediction and actual stages of each patient
    each_y_true = []
    each_y_pred = []

    for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=inputs,
                                                          targets=targets,
                                                          batch_size=network.batch_size,
                                                          seq_length=network.seq_length):
        # add arbitary noise to the input
        #x_batch = x_batch + np.random.normal(0, sigma)
        feed_dict = {
            network.input_var: x_batch,
            network.target_var: y_batch,
            #network.sigma: sigma
        }

        for i, (c, h) in enumerate(network.fw_initial_state):
            feed_dict[c] = fw_state[i].c
            feed_dict[h] = fw_state[i].h

        for i, (c, h) in enumerate(network.bw_initial_state):
            feed_dict[c] = bw_state[i].c
            feed_dict[h] = bw_state[i].h

        _, loss_value, y_pred, fw_state, bw_state = sess.run(
            [train_op, network.loss_op, network.pred_op, network.fw_final_state, network.bw_final_state],
            feed_dict=feed_dict
        )

        each_y_true.extend(y_batch)
        each_y_pred.extend(y_pred)

        total_loss += loss_value
        n_batches += 1

        # Check the loss value
        assert not np.isnan(loss_value), \
            "Model diverged with loss = NaN"

    all_fw_memory_cells.append(fw_memory_cells)
    all_bw_memory_cells.append(bw_memory_cells)
    y.append(each_y_pred)
    y_true.append(each_y_true)
    # Save memory cells and predictions
    save_dict = {
        "fw_memory_cells": fw_memory_cells,
        "bw_memory_cells": bw_memory_cells,
        "y_true": y_true,
        "y_pred": y
    }

    duration = time.time() - start_time
    total_loss /= n_batches
    total_y_pred = np.hstack(y)
    total_y_true = np.hstack(y_true)

    return total_y_true, total_y_pred, total_loss, duration

def loadEEG(uploaded_eeg):

    temp_dir_ = tempfile.TemporaryDirectory(prefix='DD-')
    temp_dir = os.path.join(temp_dir_.name, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    input_eeg = tempfile.NamedTemporaryFile(delete=False)

    if uploaded_eeg is not None:  # run only when user uploads video
        input_eeg.write(uploaded_eeg.read())

    x, y, sampling_rate = load_npz_file(input_eeg.name)

    return x, y, sampling_rate


def predict( x,y, sampling_rate, print_confusion_matrix=True, print_hypnogram = True, print_eeg=None, start_time = None, end_time=None):
    predict_start=time.time()
    # Ground truth and predictions
    y_true = []
    y_pred = []

    # The model will be built into the default Graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # Build the network
        valid_net = DeepSleepNet(
            batch_size=1,
            input_dims=30 * 100,
            n_classes=5,
            seq_length=25,
            n_rnn_layers=2,
            return_last=False,
            is_train=False,
            reuse_params=False,
            use_dropout_feature=True,
            use_dropout_sequence=True
        )

        # Initialize parameters
        valid_net.init_ops()

        checkpoint_path = os.path.join( '../model/fold0', "deepsleepnet"
        )
        # Restore the trained model
        saver = tf.train.Saver()
        # loadModel(sess, checkpoint_path)
        try:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            print("Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path)))
        except:
            model_path = os.path.join(checkpoint_path, "params_fold0.npz")
            source_model_name = 'deepsleepnet'
            # Load target model
            with np.load(model_path) as f:
                for k, v in f.iteritems():
                    if "Adam" in k or "softmax" in k or "power" in k or "global_step" in k:
                        if 'softmax' in k and 'Adam' not in k:
                            pass
                        else:
                            continue
                    prev_k = k
                    k = k.replace(source_model_name, valid_net.name)
                    tmp_tensor = tf.get_default_graph().get_tensor_by_name(k)
                    sess.run(
                        tf.assign(
                            tmp_tensor,
                            v
                        )
                    )

        print("Predicting ...\n")

        # Evaluate the model on the subject data
        y_true_, y_pred_, loss, duration = \
            custom_run_epoch(
                sess=sess, network=valid_net,
                inputs=x, targets=y,
                train_op=tf.no_op(),
                is_train=False,
            )

        n_examples = len(y_true_)
        cm_ = confusion_matrix(y_true_, y_pred_)
        acc_ = np.mean(y_true_ == y_pred_)
        mf1_ = f1_score(y_true_, y_pred_, average="macro")
        # Report performance
        print_performance(
            sess, valid_net.name,
            n_examples, duration, loss,
            cm_, acc_, mf1_
        )

        y_true.extend(y_true_)
        y_pred.extend(y_pred_)

    # Overall performance
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_examples = len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    kappa =  cohen_kappa_score(y_true, y_pred)

    #print out
    st.subheader("Overall prediction performance")
    st.markdown("Sampling rate= {:d} Hz,  # of 30-s epochs ={} ({:.3f} hours),".format(
            int(sampling_rate), n_examples, n_examples * 30/3600
        ))
    st.markdown("Prediction time: {:.2f} seconds".format(time.time() - predict_start))

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", '{:.1f} %'.format(acc*100))
    col2.metric("MF1 score", '{:.3f}'.format(mf1))
    col3.metric("Cohen's kappa", '{:.3f}'.format(kappa))

    col1, col2, col3 = st.tabs(['Confusion matrix', 'Hypnogram', 'EEG'])
    if print_confusion_matrix:
        with col1:
            st.markdown('Confusion matrix (full)')
            plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            #st.pyplot(cm_img)
    if print_hypnogram:
        with col2:

            st.markdown('Hypnogram (full)')
            hypno = plot_hypnogram(y_true, y_pred,
                           title=None, c='brown'
                           )
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

    if print_eeg:
        with col3:
            st.markdown(
                'EEG data and predicted stages from {} to {}'.format(start_time.strftime("%H:%M"),
                                                                                          end_time.strftime("%H:%M")))
            if start_time != end_time:

                def timetointeger(dtime):
                    index =  dtime.hour * 3600 + dtime.minute * 60 # in second
                    index = index /30 # split in 30 seconds
                    return int(index)

                min_dur_ind  = timetointeger(start_time)
                max_dur_ind  = timetointeger(end_time)

                plot_data = pd.DataFrame({
                    "Amplitude": x[min_dur_ind:max_dur_ind].flatten(),
                    "Time (h)": np.arange(min_dur_ind*x.shape[1], max_dur_ind*x.shape[1])/3600/sampling_rate
                    })

                line_chart = alt.Chart(plot_data).mark_line().encode(
                y= "Amplitude",
                x= "Time (h)",
                )
                classes = np.asarray(['W', 'N1', 'N2', 'N3', 'REM'])

                annotations_data = pd.DataFrame({
                    "prediction": classes[y_pred[min_dur_ind:max_dur_ind].flatten()],
                    "Time (h)":np.arange(min_dur_ind*x.shape[1]+int(x.shape[1]/2), max_dur_ind*x.shape[1]+int(x.shape[1]/2), x.shape[1])/3600/sampling_rate,
                    "Predicted stage": np.ones(len(y_pred[min_dur_ind:max_dur_ind].flatten()))*x[min_dur_ind:max_dur_ind].flatten().min() })

                annotation_layer = alt.Chart(annotations_data).mark_text(size=10,  dx=-8, dy=-10, align="left").encode(
                                        x="Time (h)",
                                        y="Predicted stage",
                                        text = alt.Text("prediction")
                                    )

                st.altair_chart((line_chart+annotation_layer).interactive(), use_container_width=True)
            else:
                st.error('Select valid time duration!')

    return acc, mf1, cm


if __name__ == '__main__':
    def start():
        x, y, sampling_rate = loadEEG(uploaded_eeg)
        max_hour, max_min = divmod(len(x)* 30/60, 60)
        start_time =None
        end_time = None

        with st.sidebar:
            st.header('Choose print options')
            print_confusion_matrix = st.checkbox('Confusion matrix (full)')
            print_hypnogram = st.checkbox('Hypnogram (full)')
            print_eeg = st.checkbox('EEG signal')
            if print_eeg:
                st.subheader('Select time duration to print EEG signal')
                st.write('Start time (0:00~)')

                col1, col2= st.columns(2)
                with col1:
                    start_hour = st.number_input('start time (h)',min_value=0, max_value=int(max_hour), format ='%02d')
                with col2:
                    start_min = st.number_input('start time (m)',min_value=0,max_value=59,format = '%02d')

                st.write('End time (~{}:{:02d})'.format(int(max_hour), int(max_min)))
                col1, col2= st.columns(2)
                with col1:
                    end_hour = st.number_input('end time (h)',min_value=0, max_value=int(max_hour), format = '%02d')
                with col2:
                    end_min = st.number_input('end time (m)',min_value=0,max_value=59, format = '%02d')

                start_time= datetime.datetime(2000,1,1,int(start_hour), int(start_min))
                end_time= datetime.datetime(2000,1,1,int(end_hour),int(end_min))
                if ((end_time - start_time).days < 0) or (end_time-start_time==datetime.timedelta(0)):
                    st.error('Select valid time duration!')

            predict_button = st.button("Predict", key='predict_button')


        if predict_button:
            predict( x, y, sampling_rate, print_confusion_matrix, print_hypnogram, print_eeg, start_time.time(), end_time.time())
    footer()

    st.title('Humanplus')
    st.subheader("Deep learning based automatic sleep staging :sleeping:")
    with st.sidebar:
        st.header('EEG file upload')
        uploaded_eeg = st.file_uploader("Upload EEG")

    #st.markdown('The data used for this project - [Sleep-EDF Database Expanded](https://www.physionet.org/content/sleep-edfx/1.0.0/) ')
    #st.markdown('Created by - [ICP lab](https://sagittak.wixsite.com/icplab)')
    #st.markdown('Github Repository - [humanplus](https://github.com/chwaaaa/humanplus)')

    if uploaded_eeg is not None:
        start()
