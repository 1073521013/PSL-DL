import tensorflow as tf
from attention import attention
import numpy as np
class CNN_LSTM(object):
    def __init__(self, sequence_length,embedding_size, filter_sizes, num_filters, num_hidden):
        num_classes=10
        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length,embedding_size], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout
        self.h_drop_input = tf.nn.dropout(self.input_x, 0.8)
        print(self.h_drop_input)
        def length(sequence):
          used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
          length = tf.reduce_sum(used, 1)
          length = tf.cast(length, tf.int32)
          return length
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
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #h = tf.layers.batch_normalization(h,training=True)
                # MAXPOOLING
#                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name="pool")
#                print(pooled)
                pooled_outputs.append(h)
        # COMBINING POOLED FEATURES
        self.h_pool = tf.concat(pooled_outputs, 2)
        print(self.h_pool)
        filter_shape = [3,120,128]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
        b = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")        
        conv2 = tf.nn.conv1d(self.h_pool, W,stride = 1,padding="SAME",name="conv2")
        h = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu")
#        with tf.name_scope('self_attention'):
#            encoder = onmt.encoders.self_attention_encoder.SelfAttentionEncoder(num_layers=1,num_units=120)  
#            outputs, state, encoded_length = encoder.encode(self.h_pool, sequence_length=length(self.input_x))
#        print(outputs)
        #3. DROPOUT LAYER ###################################################################
        with tf.name_scope("dropout_hid"):
             #self.h_drop = tf.layers.batch_normalization(self.h_pool)
             self.h_drop = tf.nn.dropout(h, self.dropout_keep_prob)
             print(self.h_drop)
        #4. LSTM LAYER ######################################################################
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)
        val_,state =  tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,self.h_drop,sequence_length=length(self.input_x),dtype=tf.float32)
        val=tf.concat(val_, 2)
        print(val)
        #embed()
        #Attention layer

        with tf.name_scope('Attention_layer'):
            attention_output, alphas = attention(val, num_hidden, return_alphas=True)
            tf.summary.histogram('alphas', alphas)

        drop = tf.nn.dropout(attention_output, self.dropout_keep_prob)
        denses = tf.layers.dense(inputs=tf.reshape(drop, shape=[-1,num_hidden*2]), units=num_hidden,activation=tf.nn.relu, trainable=True)
        denses = tf.nn.dropout(denses, self.dropout_keep_prob)
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
