import tensorflow as tf
from attention import attention, Self_Attention
import numpy as np
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from models import BaseModel, AttentionCell, highway_network, BiRNN, DenselyConnectedBiRNN

class CNN_LSTM(object):
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, num_hidden, n):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size],
                                      name="input_x")  # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")  # Dropout
        self.h_drop_input = tf.nn.dropout(self.input_x, 0.8)
        self.training = tf.placeholder(tf.bool)
        self.h_expanded = tf.expand_dims(self.h_drop_input, -1)

        print(self.h_drop_input)

        def length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
            length = tf.reduce_sum(used, 1)
            length = tf.cast(length, tf.int32)
            return length
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # CONVOLUTION LAYER
                filter_shape = [filter_size, embedding_size, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv1d(self.h_drop_input, W, stride=1, padding="SAME", name="conv")
                # conv = bn(conv,self.training)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled_outputs.append(h)
        h = tf.concat(pooled_outputs, 2)
        print(h)
        # 3. DROPOUT LAYER ###################################################################
        with tf.name_scope("dropout_hid"):
            self.h_drop = tf.nn.dropout(h, self.dropout_keep_prob)
        # 4. LSTM LAYER ######################################################################
        # cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.h_drop, sequence_length=length(self.input_x),
        #                                          dtype=tf.float32)
        # val = tf.concat(out, 2)
        # print(val)
        dense_bi_rnn = DenselyConnectedBiRNN(2, 120)
        context = dense_bi_rnn(self.h_drop, sequence_length=length(self.input_x))
        print(context)
        self.semantic = tf.nn.dropout(context, self.dropout_keep_prob)
        with tf.name_scope('Attention_layer'):
            self.val, alphas = attention(self.semantic, num_hidden, return_alphas=True)
            tf.summary.histogram('alphas', alphas)

        denses = tf.layers.dense(inputs=tf.reshape(self.val, shape=[-1, num_hidden * 2]), units=num_hidden,
                                 activation=tf.nn.relu, trainable=True)
        denses = tf.nn.dropout(denses, self.dropout_keep_prob)
        print(denses)
        #        val2 = tf.transpose(val, [1, 0, 2])
        #        last = tf.gather(val2, int(val2.get_shape()[0]) - 1)
        #        print(last)
        out_weight = tf.Variable(tf.random_normal([num_hidden, num_classes]))
        out_bias = tf.Variable(tf.random_normal([num_classes]))

        with tf.name_scope("output"):
            # lstm_final_output = val[-1]
            # embed()
            self.scores = tf.nn.xw_plus_b(denses, out_weight, out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss")

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"), name="accuracy")

        print("(!) LOADED CNN-LSTM! :)")
        # embed()
        total_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Total number of trainable parameters: %d" % total_parameters)

