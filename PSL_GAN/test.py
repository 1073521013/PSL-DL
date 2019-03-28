#! /usr/bin/env python
import sys
import codecs
# SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM  # OPTION 0
from keras.utils import np_utils
from confusionmatrix import ConfusionMatrix
from metrics_mc import *

MODEL_TO_RUN = 0
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import os
import time
import datetime
# istrain = False
# Parameters
# ==================================================
# Model Hyperparameters
embedding_dim = 21  # 128
max_seq_legth = 50
filter_sizes = [1, 3, 5, 9, 15]  # 3
num_filters = 10
dropout_prob = 0.4  # 0.5
l2_reg_lambda = 0.0
num_hidden = 32
# Training parameters
batch_size = 128
num_epochs = 50  # 200
evaluate_every = 100  # 100
checkpoint_every = 100000  # 100
num_checkpoints = 0  # Checkpoints to store
num_classes = 2
# Misc Parameters
allow_soft_placement = True
log_device_placement = False
max_sequence_size = 50
X = []
# Data Preparation
with open("save/generator.txt") as f: 
    for sequence in f:
        sequence = sequence.strip()
        sequence = sequence.split(' ')
        for i,t in enumerate(sequence):
            sequence[i] = str(chr(int(t)))
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
                sequence.pop(26)
            # sequences.append(list((ord(t)-64) for t in sequence))
            sequences.append(sequence)
        else:
            b = int((max_sequence_size - len(sequence)))
            for i in list(range(b)):
                sequence.insert(int((len(sequence))), '0')
            sequences.append(sequence)
        X.append(sequences[0])
acid_letters = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']
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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
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
        checkpoint_dir = os.path.abspath(os.path.join("runs", "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        # EVALUATE MODEL
        def dev_step(x_batch, writer=None, save=False):
            feed_dict = {model.input_x: x_batch, model.dropout_keep_prob: 0.5}
            step, pre= sess.run([global_step, model.predictions], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}".format(time_str, step))
            if save:
                if writer:
                    writer.add_summary(summaries, step)
            return pre

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
with codecs.open("save/flit.txt", "w","utf8") as f2:
    predictions = dev_step(x_train)
    outputs = np.argmax(predictions,axis=-1)
    for i in outputs:
    	f2.write(str(i))
    	f2.write('\n')

