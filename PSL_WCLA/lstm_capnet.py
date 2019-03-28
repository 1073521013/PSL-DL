import tensorflow as tf
from attention import attention,Self_Attention
import opennmt as onmt
import numpy as np
#from attention_all import self_align_attention
from tcn import TemporalConvNet
from capsLayer import conv_caps_layer, fully_connected_caps_layer
class LSTM_CNN(object):
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, num_hidden):
        l2_reg_lambda=0.0
        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout
        self.training = tf.placeholder(tf.bool)
        with tf.name_scope("dropout1"):
            self.input_x_ = tf.nn.dropout(self.input_x, 0.8)
        
        l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss

        #1. EMBEDDING LAYER ################################################################
#        with tf.device('/cpu:0'), tf.name_scope("embedding"):
#            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
#            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        def length(sequence):
          used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
          length = tf.reduce_sum(used, 1)
          length = tf.cast(length, tf.int32)
          return length
        #2. LSTM LAYER ######################################################################       
        self.cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)
        self.cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)        
        self.lstm_out,_ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,self.input_x_,sequence_length=length(self.input_x),dtype=tf.float32)
        #self.lstm_out,_ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,self.input_x_,dtype=tf.float32)
        self.lstm_out=tf.concat(self.lstm_out, 2)
#        with tf.variable_scope("self_attention"):
#            val = multihead_attention(y,None,None,num_hidden,num_hidden,num_hidden,4,0.9)
#            print(val)  
        print(self.lstm_out)
#        num_chans = [100] * (3 - 1) + [num_hidden]
#        self.tcn = TemporalConvNet(num_chans, stride=1, kernel_size=2, dropout=0.2)
#        self.semantic = self.tcn(self.lstm_out)
#        print(self.semantic)
        self.lstm_out = tf.nn.dropout(self.lstm_out, self.dropout_keep_prob)        
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # CONVOLUTION LAYER
                filter_shape = [filter_size, embedding_size,  num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                print(W)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv1d(self.lstm_out, W,stride = 1,padding="SAME",name="conv")
                print(conv)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #h = tf.layers.batch_normalization(h,training=True)
                # MAXPOOLING
#                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name="pool")
#                print(pooled)
                pooled_outputs.append(h)
        # COMBINING POOLED FEATURES
        h = tf.concat(pooled_outputs, 2)
        h = tf.nn.dropout(h, self.dropout_keep_prob)
        filter_shape = [3, 120, 120]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
        b = tf.Variable(tf.constant(0.1, shape=[120]), name="b2")
        conv2 = tf.nn.conv1d(h, W, stride=1, padding="SAME", name="conv2")
        h = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu")

#        with tf.name_scope('self_attention'):
#            encoder = onmt.encoders.self_attention_encoder.SelfAttentionEncoder(num_layers=1,num_units=120)  
#            self.semantic, state, encoded_length = encoder.encode(self.semantic, sequence_length=length(self.input_x))
        #conv1 = tf.expand_dims(h, -1)
        #print(conv1)        
        with tf.variable_scope('PrimaryCaps_layer'):
            cap1 = conv_caps_layer(input_layer=h,capsules_size=120, nb_filters=2, kernel=8)
        print(cap1)
        # DigitCaps layer, return shape [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            cap2 = fully_connected_caps_layer(input_layer=cap1,capsules_size=300, nb_capsules=2)
        print(cap2)

        #self.caps = tf.squeeze(caps1) 
        self.semantic = tf.nn.dropout(cap2, self.dropout_keep_prob)
        
        
        # #3. DROPOUT LAYER ###################################################################
        with tf.name_scope('Attention_layer'):
            self.val, alphas = attention(self.semantic, 120, return_alphas=True)
            tf.summary.histogram('alphas', alphas)
         
        #self.val2, _ = attention(self.lstm_out, 100, return_alphas=True)        
        #self.val = tf.concat([self.val1,self.val2],axis=1)
        #print(self.val)
        drop = tf.nn.dropout(self.val, self.dropout_keep_prob)
        denses = tf.layers.dense(inputs=tf.reshape(drop, shape=[-1,120]), units=60,activation=tf.nn.relu, trainable=True)
        out_weight = tf.Variable(tf.random_normal([60, num_classes]))
        out_bias = tf.Variable(tf.random_normal([num_classes]))        
        
        with tf.name_scope("output"):
            self.scores = tf.nn.xw_plus_b(denses, out_weight,out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")
            #self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            #correct_predictions = tf.equal(tf.argmax(self.predictions,1), tf.argmax(self.input_y, 1))
            correct_predictions = tf.equal(tf.argmax(self.scores, 1, name="predictions"), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


        print ("(!!) LOADED LSTM-CNN! :)")
        #embed()

        total_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Total number of trainable parameters: %d" % total_parameters)

# 1. Embed --> LSTM
# 2. LSTM --> CNN
# 3. CNN --> Pooling/Output
