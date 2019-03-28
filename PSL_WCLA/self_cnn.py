import tensorflow as tf
from attention import attention,Self_Attention
import opennmt as onmt
import numpy as np
from ops.attention import multihead_attention
class CNN_LSTM(object):
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, num_hidden):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length,embedding_size], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout
        self.h_drop_input = tf.nn.dropout(self.input_x, 0.7)
        self.training = tf.placeholder(tf.bool)
        print(self.h_drop_input)
        def length(sequence):
          used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
          length = tf.reduce_sum(used, 1)
          length = tf.cast(length, tf.int32)
          return length
        def bn(x, train_phase):
            with tf.variable_scope('bn'):
                beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
                axises = list(np.arange(len(x.shape) - 1))
                batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
                ema = tf.train.ExponentialMovingAverage(decay=0.5)
        
                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)
        
                mean, var = tf.cond(train_phase, mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
            return normed

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
                conv = bn(conv,self.training)                
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #h = tf.layers.batch_normalization(h,training=True)
                # MAXPOOLING
#                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name="pool")
#                print(pooled)
                pooled_outputs.append(h)
        # COMBINING POOLED FEATURES
        h = tf.concat(pooled_outputs, 2)
        print(h)
        filter_shape = [3, 90, 128]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
        b = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")        
        conv2 = tf.nn.conv1d(h, W,stride = 1,padding="SAME",name="conv2")
        conv2= bn(conv2, self.training)
        y = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu")
#        self.cell_fw = tf.nn.rnn_cell.BasicLSTMCell(100,forget_bias=1.0)
#        self.cell_bw = tf.nn.rnn_cell.BasicLSTMCell(100,forget_bias=1.0)        
#        self.lstm_out,_ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,h,sequence_length=length(self.input_x),dtype=tf.float32)
#        #self.lstm_out,_ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,self.input_x_,dtype=tf.float32)
#        y=tf.concat(self.lstm_out, 2)        
     
        with tf.name_scope('self_attention'):
            encoder = onmt.encoders.self_attention_encoder.SelfAttentionEncoder(num_layers=1,num_units=128)  
            self.semantic, state, encoded_length = encoder.encode(h, sequence_length=length(self.input_x))
        print(self.semantic)
#        with tf.variable_scope("self_attention"):
#            self.semantic = multihead_attention(y,None,None,120,120,120,10,0.9)
#            print(self.semantic)
#        with tf.name_scope('self_attention'):
#            y=Self_Attention(y,y,y,4,50)
#            print(y)
        #3. DROPOUT LAYER ###################################################################
        with tf.name_scope("dropout_hid"):
             #self.h_drop = tf.layers.batch_normalization(self.h_pool)
             self.h_drop = tf.nn.dropout(self.semantic, self.dropout_keep_prob)
             print(self.h_drop)   
        #embed()
        #Attention layer
#        with tf.name_scope('self_attention'):
#            encoder = onmt.encoders.self_attention_encoder.SelfAttentionEncoder(num_layers=1,num_units=256)  
#            outputs, state, encoded_length = encoder.encode(val, sequence_length=length(self.input_x))
#        print(outputs)
        with tf.name_scope('Attention_layer'):
            self.val, alphas = attention(self.h_drop, num_hidden, return_alphas=True)
            tf.summary.histogram('alphas', alphas)

        drop = tf.nn.dropout(self.val, self.dropout_keep_prob)
        denses = tf.layers.dense(inputs=tf.reshape(drop, shape=[-1,num_hidden]), units=num_hidden/2,activation=tf.nn.relu, trainable=True)
        denses = tf.nn.dropout(denses, self.dropout_keep_prob)
#        val2 = tf.transpose(val, [1, 0, 2])
#        last = tf.gather(val2, int(val2.get_shape()[0]) - 1) 
#        print(last)
        out_weight = tf.Variable(tf.random_normal([int(num_hidden/2), num_classes]))
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
