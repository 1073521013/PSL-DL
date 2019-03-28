import tensorflow as tf
from attention import attention,Self_Attention
import numpy as np
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

class CNN_LSTM(object):
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, num_hidden,n):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length,embedding_size], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout
        self.h_drop_input = tf.nn.dropout(self.input_x, 0.8)
        self.training = tf.placeholder(tf.bool)
        # self.h_expanded = tf.expand_dims(self.h_drop_input, -1)
	
        print(self.h_drop_input)
        def length(sequence):
          used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
          length = tf.reduce_sum(used, 1)
          length = tf.cast(length, tf.int32)
          return length

        '''
        def conv(name, l, channel, stride):
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))
        def add_layer(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                #c = BatchNorm('bn1', l)
                c = tf.nn.relu(l)
                c = conv('conv1', c, 12, 1)
                l = tf.concat([c, l], 3)
            return l
        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                #l = BatchNorm('bn1', l)
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
                l = AvgPooling('pool', l, 2)
            return l
        def dense_net(name):
            l = conv('conv0', self.h_expanded, 16, 1)
            with tf.variable_scope('block1') as scope:
                for i in range(n):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition1', l)
            with tf.variable_scope('block2') as scope:
                for i in range(n):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition2', l)
            with tf.variable_scope('block3') as scope:
                for i in range(n):
                    l = add_layer('dense_layer.{}'.format(i), l)
            #l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)  ##shape=(?, 250, 5, 88)
            #l = GlobalAvgPooling('gap', l)
            #logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)
            l = tf.reshape(l, [-1, 250*5 , 88])
            return l
        logits = dense_net("dense_net")
        print(logits)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # CONVOLUTION LAYER
                filter_shape = [filter_size, embedding_size,  num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv1d(self.h_drop_input, W,stride = 1,padding="SAME",name="conv")
                #conv = bn(conv,self.training)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled_outputs.append(h)
        h = tf.concat(pooled_outputs, 2)
        print(h)
        #3. DROPOUT LAYER ###################################################################
        with tf.name_scope("dropout_hid"):
             #self.h_drop = tf.layers.batch_normalization(self.h_pool)
             self.h_drop = tf.nn.dropout(logits, self.dropout_keep_prob)
        #4. LSTM LAYER ######################################################################
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)      
        out,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,self.h_drop,sequence_length=length(self.input_x),dtype=tf.float32)
        #self.lstm_out,_ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,self.input_x_,dtype=tf.float32)
        val=tf.concat(out, 2)
        print(val)
        self.semantic = tf.nn.dropout(val, self.dropout_keep_prob)
        with tf.name_scope('Attention_layer'):
            self.val, alphas = attention(self.semantic, num_hidden, return_alphas=True)
            tf.summary.histogram('alphas', alphas)
        denses = tf.layers.dense(inputs=tf.reshape(self.val, shape=[-1,num_hidden*2]), units=num_hidden,activation=tf.nn.relu, trainable=True)
        denses = tf.nn.dropout(denses, self.dropout_keep_prob)
        '''


        def conv(l):
            shape = l.get_shape().as_list()
            in_channel = shape[2]
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # CONVOLUTION LAYER
                    filter_shape = [filter_size, in_channel, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv_ = tf.nn.conv1d(l, W, stride=1, padding="SAME", name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv_, b), name="relu")
                    pooled_outputs.append(h)
            # COMBINING POOLED FEATURES
            h = tf.concat(pooled_outputs, 2)
            return h


        def add_transition(l):
            shape = l.get_shape().as_list()
            in_channel = shape[2]
            filter_shape = [3, in_channel, in_channel]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
            b = tf.Variable(tf.constant(0.1, shape=[in_channel]), name="b2")
            conv2 = tf.nn.conv1d(l, W, stride=1, padding="SAME", name="conv2")
            y = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu")
            return y


        def add_layer(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[2]
            with tf.variable_scope(name) as scope:
                l = tf.nn.relu(l)
                c = conv(l)
                l = tf.concat([c, l], 2)
                print(name)
                print(l)
            return l


        def dense_net(name):
            l = conv(self.h_drop_input)
            with tf.variable_scope('block1') as scope:
                for i in range(n):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition(l)
            with tf.variable_scope('block2') as scope:
                for i in range(n):
                    l = add_layer('dense_layer.{}'.format(i), l)
            l = tf.nn.relu(l)  ##shape=(?, 250, 5, 88)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)
            # l = tf.reshape(l, [-1, 250 * 5, 88])
            return l
        logits = dense_net("dense_net")
        print(logits)
        denses= tf.nn.dropout(logits, self.dropout_keep_prob)
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

