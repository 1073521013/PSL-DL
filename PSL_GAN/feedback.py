#! /usr/bin/env python
from __future__ import division
import sys
import codecs
# SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM  # OPTION 0
from keras.utils import np_utils
from confusionmatrix import ConfusionMatrix
from metrics_mc import *
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import batchgen
# istrain = False
# Parameters
# ==================================================
# Model Hyperparameters
embedding_dim = 21  # 128
filter_sizes = [1, 3, 5, 9, 15,21]  # 3
num_filters = 20
dropout_prob = 0.5  # 0.5
l2_reg_lambda = 0.0
num_hidden = 60
# Training parameters
batch_size = 200
num_epochs = 1  # 200
evaluate_every = 100  # 100
checkpoint_every = 100000  # 100
num_checkpoints = 0  # Checkpoints to store
num_classes = 10
# Misc Parameters
allow_soft_placement = True
log_device_placement = False
max_sequence_size = 100

word2idx = np.load("save/w2i.npy")
word2idx = word2idx.item()
id2word = {k: v for v, k in zip(word2idx.keys(), word2idx.values())}
print(id2word)
def pre(seq, value):
    global pres0, pres
    X = []
    for sequence in seq:
        sequence = sequence.tolist()
        for i, t in enumerate(sequence):
            t = id2word.get((int(t)))
            if t is None:
                sequence[i] = '0'
            else:
                sequence[i] = t
        sequences = []
        sequence = ''.join(sequence)
        sequence = sequence.replace('X', '0')
        sequence = sequence.replace('U', '0')
        sequence = sequence.replace('O', '0')
        sequence = sequence.replace('B', 'N')
        sequence = sequence.replace('Z', 'Q')
        sequence = sequence.replace('J', 'L')
        sequence = list(sequence)

        if len(sequence) >= max_sequence_size:
            a = int((len(sequence) - max_sequence_size))
            for i in list(range(a)):
                sequence.pop(51)
            # sequences.append(list((ord(t)-64) for t in sequence))
            sequences.append(sequence)
        else:
            b = int((max_sequence_size - len(sequence)))
            for i in list(range(b)):
                sequence.insert(int((len(sequence))), '0')
            sequences.append(sequence)
        X.append(sequences[0])
    acid_letters = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V',
                    'Y']
    le = LabelEncoder()
    datas = np_utils.to_categorical(le.fit_transform(acid_letters))

    def two2three(x):
        xx = []
        for _, m in enumerate(x):
            k = []
            for j, t in enumerate(m):
                if t not in acid_letters:
                    t = '0'
                n = acid_letters.index(t)
                k.append(datas[n])
            xx.append(k)
        return np.array(xx)
    x_train = np.array(two2three(X))
    print(x_train.shape)
    # Training
    # ==================================================
    batches = batchgen.gen_batch(x_train, batch_size, num_epochs)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = CNN_LSTM(x_train.shape[1], embedding_dim, filter_sizes, num_filters, num_hidden)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join("runs_deeploc", "checkpoints"))
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
            sess.run(tf.global_variables_initializer())

            # EVALUATE MODEL
            def dev_step(x_batch, writer=None):
                feed_dict = {model.input_x: x_batch, model.dropout_keep_prob: 1}
                pre = sess.run(model.predictions, feed_dict)
                return pre

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
    new_posi = []
    new_nege = []
    '''
    for i,batch in enumerate(batches):
        if i == 0:
            pres0 = dev_step(batch)
        elif i == 1:
            pres1 = dev_step(batch)
            pres = np.concatenate((pres0, pres1))
        else:
            press = dev_step(batch)
            pres = np.concatenate((pres, press))
    '''
    pres = dev_step(x_train)
    # ['Cell.membrane', 'Cytoplasm', 'Endoplasmic.reticulum', 'Extracellular', 'Golgi apparatus', 'Lysosome/Vacuole', 'Mitochandrion', 'Nucleus', 'Peroxisome', 'Plastid']
    # ['Cell.membrane', 'Cytoplasm', 'Endoplasmic.reticulum', 'Golgi.apparatus', 'Lysosome/Vacuole', 'Mitochondrion', 'Nucleus', 'Peroxisome', 'Plastid', 'Extracellular']     [0, 1, 2, 4, 5, 6, 7, 8, 9, 3]
    print(len(pres))
    for i, t in enumerate(pres):
        #print(np.argmax(t))
        if t[7] >= value:
            new_posi.append(seq[i])
        elif t[3] >= value:
            new_nege.append(seq[i])
    print(len(new_posi))
    accuracy = len(new_posi) / len(pres)
    print(accuracy)
    with open("acc_0.65.txt", "a") as f:
        f.write(str(accuracy))
        f.write("\n")
    return new_posi,new_nege
