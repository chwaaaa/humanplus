
import streamlit as st
import itertools

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import pandas as pd
import altair as alt
import numpy as np
from io import BytesIO



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

    fig, ax = plt.subplots(2,1, figsize=(10, 5))
    c = 'brown'
    x_axis = np.arange(y_pred.shape[0])
    ax[0].plot(x_axis, y_true, color=c)

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
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

def plot_graph(x, y_true, y_pred, min_dur_ind, max_dur_ind, sampling_rate):
    plot_data = pd.DataFrame({
        "Amplitude": x[min_dur_ind:max_dur_ind].flatten(),
        "Time (h)": np.arange(min_dur_ind * x.shape[1], max_dur_ind * x.shape[1]) / 3600 / sampling_rate
    })

    line_chart = alt.Chart(plot_data).mark_line(color="black").encode(
        y=alt.Y("Amplitude",title='Amplitude'),
        x="Time (h)",
    )
    classes = np.asarray(['Wake', 'N1', 'N2', 'N3', 'REM'],dtype='str')

    true_annotations_data = pd.DataFrame({
        "prediction": classes[y_true[min_dur_ind:max_dur_ind].flatten()],
        "Time (h)": np.arange(min_dur_ind * x.shape[1] + int(x.shape[1] / 2),
                              max_dur_ind * x.shape[1] + int(x.shape[1] / 2), x.shape[1]) / 3600 / sampling_rate,
        "Labeled stage": np.ones(len(y_pred[min_dur_ind:max_dur_ind].flatten())) * x[
                                                                                     min_dur_ind:max_dur_ind].flatten().min()-20,
        "Label": np.tile(np.array(['True']),len(y_pred[min_dur_ind:max_dur_ind].flatten()))
    })

    true_annotation_layer = alt.Chart(true_annotations_data).mark_text(size=13, dx=-8, dy=-10, align="center").encode(
        x="Time (h)",
        y=alt.Y("Labeled stage"),
        text=alt.Text("prediction"),
        color="Label"
    )

    pred_annotations_data = pd.DataFrame({
         "prediction": classes[y_pred[min_dur_ind:max_dur_ind].flatten()],
         "Time (h)": np.arange(min_dur_ind * x.shape[1] + int(x.shape[1] / 2),
                               max_dur_ind * x.shape[1] + int(x.shape[1] / 2), x.shape[1]) / 3600 / sampling_rate,
         "Predicted stage": np.ones(len(y_pred[min_dur_ind:max_dur_ind].flatten())) * x[
                                                                                      min_dur_ind:max_dur_ind].flatten().min(),
         "Label": np.tile(np.array(['Prediction']),len(y_pred[min_dur_ind:max_dur_ind].flatten()))})

    pred_annotation_layer = alt.Chart(pred_annotations_data).mark_text(size=13, dx=-8, dy=-10, align="center").encode(
        x="Time (h)",
        y=alt.Y("Predicted stage"),
        text=alt.Text("prediction"),
        color="Label"
    )

    st.altair_chart((line_chart + pred_annotation_layer + true_annotation_layer).interactive(), use_container_width=True)



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


