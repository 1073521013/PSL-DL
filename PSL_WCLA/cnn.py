import tensorflow as tf
from attention import attention,Self_Attention
#import opennmt as onmt
#from ops.attention import multihead_attention
from model_components import task_specific_attention, bidirectional_rnn
from lstm import  BNLSTMCell
from bn_layer.rnn_cell_layernorm_modern import BasicLSTMCell_LayerNorm
from tensorflow.python.training.moving_averages import assign_moving_average
import numpy as np
class CNN_LSTM(object):
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, num_hidden):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length,embedding_size], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout
        self.h_drop_input = tf.nn.dropout(self.input_x, 0.8)
        self.training = tf.placeholder(tf.bool)
        print(self.h_drop_input)
        def length(sequence):
          used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
          length = tf.reduce_sum(used, 1)
          length = tf.cast(length, tf.int32)
          return length
        def batchnorm(Ylogits, is_test, offset, convolutional=True):
            #Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
            #Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
            #Y2r = tf.nn.relu(Y2bn)
            exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, 10000) # adding the iteration prevents from averaging across non-existing iterations
            bnepsilon = 1e-5
            if convolutional:
                mean, variance = tf.nn.moments(Ylogits, [0, 1])
            else:
                mean, variance = tf.nn.moments(Ylogits, [0])
            update_moving_averages = exp_moving_avg.apply([mean, variance])
            m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
            v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
            Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
            return Ybn, update_moving_averages


        #l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss
        #1. EMBEDDING LAYER ################################################################
#        with tf.device('/cpu:0'), tf.name_scope("embedding"):
#            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
#            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
#            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #2. CONVOLUTION LAYER + MAXPOOLING LAYER (per filter) ###############################
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # CONVOLUTION LAYER
                filter_shape = [filter_size, embedding_size,  num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                print(W)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv1d(self.h_drop_input, W,stride = 1,padding="SAME",name="conv")
                print(conv)
                conv ,_= batchnorm(conv, self.training ,b)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #h = batch_norm_layer(h,train_phase=self.training,scope_bn='bn')
                # MAXPOOLING
#                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name="pool")
#                print(pooled)
                pooled_outputs.append(h)
        # COMBINING POOLED FEATURES
        h = tf.concat(pooled_outputs, 2)
        print(h)
        filter_shape = [3, 120, 128]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
        b = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")        
        conv2 = tf.nn.conv1d(h, W,stride = 1,padding="SAME",name="conv2")
        conv2 ,_= batchnorm(conv2, self.training ,b)
        y = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu")
        #y = batch_norm_layer(y,train_phase=self.training,scope_bn='bn')
        #3. DROPOUT LAYER ###################################################################
        with tf.name_scope("dropout_hid"):
             #self.h_drop = tf.layers.batch_normalization(self.h_pool)
             self.h_drop = tf.nn.dropout(y, self.dropout_keep_prob)
             print(self.h_drop)
        #4. LSTM LAYER ######################################################################
        
#        cell_fw = BNLSTMCell(num_hidden, self.training) #LSTMCell(hidden_size)
#        cell_bw = BNLSTMCell(num_hidden, self.training)
        cell_fw = BasicLSTMCell_LayerNorm(num_hidden)
        cell_bw = BasicLSTMCell_LayerNorm(num_hidden)
        val_,state = bidirectional_rnn(cell_fw ,cell_bw, self.h_drop,input_lengths=length(self.input_x))
        #_, final_hidden = state
        val=tf.concat(val_, 2)
        print(val)
        self.semantic = tf.nn.dropout(val, self.dropout_keep_prob)
        #embed()
        #Attention layer
        with tf.variable_scope("self_attention"):
            self.val = task_specific_attention(self.semantic, num_hidden*2)
            print(self.val)        
#        with tf.name_scope('self_attention'):
#            val=Self_Attention(val,val,val,2,num_hidden/2)
#            print(val)
#        with tf.name_scope('Attention_layer'):
#            attention_output, alphas = attention(val, num_hidden, return_alphas=True)
#            tf.summary.histogram('alphas', alphas)
        
        denses = tf.layers.dense(inputs=tf.reshape(self.val, shape=[-1,num_hidden*2]), units=num_hidden,activation=tf.nn.relu, trainable=True)
        denses = tf.nn.dropout(denses, self.dropout_keep_prob)
        print(denses)
#        val2 = tf.transpose(val, [1, 0, 2])
#        last = tf.gather(val2, int(val2.get_shape()[0]) - 1) 
#        print(last)
        out_weight = tf.Variable(tf.random_normal([num_hidden, num_classes]))
        out_bias = tf.Variable(tf.random_normal([num_classes]))

        with tf.name_scope("output"):
            #lstm_final_output = val[-1]
            #embed()
            self.scores = tf.nn.xw_plus_b(denses, out_weight,out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss")

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1),tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),name="accuracy")

        print ("(!) LOADED CNN-LSTM! :)")
        #embed()
        total_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Total number of trainable parameters: %d" % total_parameters)

