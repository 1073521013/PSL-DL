#! /usr/bin/env python
import sys

# SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM  # OPTION 0
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import np_utils
from confusionmatrix import ConfusionMatrix
from metrics_mc import *
from Bio import SeqIO
MODEL_TO_RUN = 0
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import batchgen

#istrain = True
istrain = False
# Parameters
# ==================================================
# Model Hyperparameters
embedding_dim = 21  # 128

filter_sizes = [1, 3, 5, 9, 15,21]  # 3
num_filters = 20
dropout_prob = 0.5 # 0.5
l2_reg_lambda = 0.0
num_hidden = 60
# Training parameters
batch_size = 200
num_epochs = 300  # 200
evaluate_every = 100  # 100
checkpoint_every = 100000  # 100
num_checkpoints = 1  # Checkpoints to store
num_classes = 10


max_sequence_size = 100
X = []
acid_letters = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']
le = LabelEncoder()
datas = np_utils.to_categorical(le.fit_transform(acid_letters))

y = []
fasta_sequences = SeqIO.parse(open('save/deeploc_data.fasta'), 'fasta')
for fasta in fasta_sequences:
    sequences = []
    name, class0 = fasta.id, fasta.description.split()[1]
    class0 = class0.split('-')[0]
    #y.append(class0)
    sequence = str(fasta.seq)
    sequence = sequence.replace('X', '0')
    sequence = sequence.replace('U', '0')
    sequence = sequence.replace('O', '0')
    sequence = sequence.replace('B', 'N')
    sequence = sequence.replace('Z', 'Q')
    sequence = sequence.replace('J', 'L')
    sequence = list(sequence)
    if len(sequence)<300:
        y.append(class0)
        if len(sequence) >= max_sequence_size:
            a = int((len(sequence) - max_sequence_size))
            for i in list(range(a)):
                sequence.pop(51)
            sequences.append(sequence)
        else:
            b = int((max_sequence_size - len(sequence)))
            for i in list(range(b)):
                sequence.insert(int((len(sequence))), '0')
            sequences.append(sequence)
        X.append(sequences[0])

print(len(X))
def two2three(x):
    xx = []
    for _, m in enumerate(x):
        k = []
        for j, t in enumerate(m):
            n = acid_letters.index(t)
            k.append(datas[n])
        xx.append(k)
    return np.array(xx)
def quchong(li):
    new_li=[]
    for i in li:
        if i not in new_li:
            new_li.append(i)
    return new_li

X = np.array(two2three(X))
le = LabelEncoder()

print(quchong(y))
print(quchong(le.fit_transform(y)))
y = np_utils.to_categorical(le.fit_transform(y))

print(X.shape)
print(y.shape)
num = X.shape[0]
indices = np.arange(num)
np.random.seed(seed=123456)
np.random.shuffle(indices)
x = X[indices]
y = y[indices]
test_index = []
train_index = []
for i in range(num):
    if i % 4 == 0:
        test_index.append(i)
    else:
        train_index.append(i)
x_test = x[test_index, :, :]
x_train = x[train_index, :, :]
y_test = y[test_index, :]
y_train = y[train_index, :]
print(x_train.shape)
print(y_train.shape)
# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # embed()
        if (MODEL_TO_RUN == 0):
            model = CNN_LSTM(x_train.shape[1], embedding_dim, filter_sizes, num_filters, num_hidden)
        elif (MODEL_TO_RUN == 1):
            model = LSTM_CNN(x_train.shape[1], y_train.shape[1], 21, embedding_dim, filter_sizes, num_filters,
                             l2_reg_lambda)
        elif (MODEL_TO_RUN == 2):
            model = CNN(x_train.shape[1], y_train.shape[1], 26, embedding_dim, filter_sizes, num_filters, l2_reg_lambda)
        elif (MODEL_TO_RUN == 3):
            model = LSTM(x_train.shape[1], y_train.shape[1], 26, embedding_dim)
        else:
            print
            "PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n"
            exit();

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_deeploc", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join("runs_deeploc", "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        # TRAINING STEP
        def train_step(x_batch, y_batch, save=False):
            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: dropout_prob}
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                train_summary_writer.add_summary(summaries, step)


        # EVALUATE MODEL
        def dev_step(x_batch, y_batch, writer=None, save=False):
            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: 0.5}
            step, pre, summaries, loss, accuracy = sess.run([global_step, model.scores, dev_summary_op, model.loss, model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                if writer:
                    writer.add_summary(summaries, step)
            return accuracy, pre


        # CREATE THE BATCHES GENERATOR
        batches = batchgen.gen_batch(list(zip(x_train, y_train)), batch_size, num_epochs)

        max_acc = 0
        if istrain:
            for i, batch in enumerate(batches):
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    accuracy, predictions = dev_step(x_test, y_test, writer=dev_summary_writer)
                    if accuracy > max_acc:
                        max_acc = accuracy
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            accuracy, predictions = dev_step(x_test, y_test)  # dev_step(x_dev, y_dev, writer=dev_summary_writer)
outputs = np.argmax(predictions, axis=-1)
print(outputs.shape)
confusion_test = ConfusionMatrix(num_classes)
y = np.argmax(y_test, axis=-1)
print(y.shape)
confusion_test.batch_add(y, outputs)
test_accuracy = confusion_test.accuracy()
cf_test = confusion_test.ret_mat()
from sklearn.metrics import classification_report
print(classification_report(y, outputs, labels=None, target_names=None, sample_weight=None, digits=2))

print("FINAL TEST RESULTS")
print(confusion_test)
print("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
print("  test Gorodkin:\t\t{:.2f}".format(gorodkin(cf_test)))
print("  test IC:\t\t{:.2f}".format(IC(cf_test)))

'''

class_names=['Cell.membrane', 'Cytoplasm', 'Endoplasmic.reticulum', 'Extracellular', "Golgi.apparatus",
           "Lysosome", "Mitochondrion", "Nucleus", "Peroxisome", "Plastid"]
"""
Compute ROC curve and ROC area for each class
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc

enc = OneHotEncoder(sparse=False)
enc.fit(y.reshape(-1, 1))
onehot = enc.transform(y.reshape(-1, 1))

plt.gca().set_color_cycle(['red', 'green', 'blue', 'peru','orange','purple','brown','pink','gray','cyan'])
fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()
co = []
for (i, class_name) in enumerate(class_names):
    fpr[class_name], tpr[class_name], thresholds[class_name] = roc_curve(onehot[:, i], predictions[:, i])
    roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])

    youdens = tpr[class_name] - fpr[class_name]
    index = np.argmax(youdens)
    # youden[class_name]=tpr[class_name](index)
    fpr_val = fpr[class_name][index]
    tpr_val = tpr[class_name][index]
    thresholds_val = thresholds[class_name][index]

    p_auto = predictions[:, i].copy()
    t_auto = onehot[:, i].copy()
    p_auto[p_auto >= thresholds_val] = 1
    p_auto[p_auto < thresholds_val] = 0
    acc = np.float(np.sum(t_auto == p_auto)) / t_auto.size
    co.append(class_name+'(AUC={:.2f})'.format(roc_auc[class_name]))
    plt.plot(fpr[class_name], tpr[class_name], lw=2, label=class_name + '(%0.2f)' % roc_auc[class_name])
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of each class')
#plt.title('ROC of %s ' % class_name + '(AUC={:.2f}, Thr={:.2}, Acc={:.2f}%'.format(roc_auc[class_name], thresholds_val,  acc * 100))
plt.legend(co)
#plt.savefig('jpg/roc_%s.jpg'%class_name)
plt.savefig('all_roc.jpg')
'''

