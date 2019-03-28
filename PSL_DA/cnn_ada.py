import tensorflow as tf
from attention import attention,Self_Attention
#import opennmt as onmt
#from ops.attention import multihead_attention
import numpy as np
from flip_gradient import flip_gradient
from model import DenselyConnectedBiRNN

class CNN_LSTM(object):
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, num_hidden,batch_size):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length,embedding_size], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout
        self.h_drop_input = tf.nn.dropout(self.input_x, 0.7)
        self.training = tf.placeholder(tf.bool)
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        with tf.variable_scope('feature_extractor'):        
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
                    #conv = bn(conv,self.training)
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled_outputs.append(h)
            # COMBINING POOLED FEATURES
            h = tf.concat(pooled_outputs, 2)
            print(h)
            filter_shape = [1, 120, 128]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
            b = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")        
            conv2 = tf.nn.conv1d(h, W,stride = 1,padding="SAME",name="conv2")
            #conv2= bn(conv2, self.training)
            y = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu")
            #y = batch_norm_layer(y,train_phase=self.training,scope_bn='bn')
            #3. DROPOUT LAYER ###################################################################
            with tf.name_scope("dropout_hid"):
                 #self.h_drop = tf.layers.batch_normalization(self.h_pool)
                 self.h_drop = tf.nn.dropout(y, self.dropout_keep_prob)
                 print(self.h_drop)
            #4. LSTM LAYER ######################################################################
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)      
            #out,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,self.h_drop,sequence_length=length(self.input_x),dtype=tf.float32)
            out,_,_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([cell_fw], [cell_bw], self.h_drop,sequence_length=length(self.input_x), dtype=tf.float32)
            val=tf.concat(out, 2)
	    #dense_bi_rnn = DenselyConnectedBiRNN(2, num_hidden)
            #val = dense_bi_rnn(self.h_drop, seq_len=length(self.input_x))
            print(val)
            self.semantic = tf.nn.dropout(val, self.dropout_keep_prob)
            with tf.name_scope('Attention_layer'):
                self.feature, alphas = attention(self.semantic, num_hidden*2, return_alphas=True)
                tf.summary.histogram('alphas', alphas)
      
        with tf.variable_scope('label_predictor'):        
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size // 2, -1])
            classify_feats = tf.cond(self.training, source_features, all_features)
            
            all_labels = lambda: self.input_y
            source_labels = lambda: tf.slice(self.input_y, [0, 0], [batch_size // 2, -1])
            self.classify_labels = tf.cond(self.training, source_labels, all_labels)
            
            
            denses = tf.layers.dense(inputs=tf.reshape(classify_feats, shape=[-1,num_hidden*2]), units=num_hidden,activation=tf.nn.relu, trainable=True)
            denses = tf.nn.dropout(denses, self.dropout_keep_prob)
            out_weight = tf.Variable(tf.random_normal([num_hidden, num_classes]))
            out_bias = tf.Variable(tf.random_normal([num_classes]))
    
            with tf.name_scope("output"):
                self.score = tf.nn.xw_plus_b(denses, out_weight,out_bias, name="scores")
                self.predictions = tf.nn.softmax(self.score, name="predictions")
    
            with tf.name_scope("loss"):
                self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score,labels=self.classify_labels)
                self.pred_loss = tf.reduce_mean(self.losses, name="loss")
    
            with tf.name_scope("accuracy"):
                self.correct_label_pred = tf.equal(tf.argmax(self.predictions, 1),tf.argmax(self.classify_labels, 1))
                self.label_acc = tf.reduce_mean(tf.cast(self.correct_label_pred, "float"),name="accuracy")
                
        with tf.variable_scope('domain_predictor'):
            feat = flip_gradient(self.feature, self.l)
            denses = tf.layers.dense(inputs=tf.reshape(feat, shape=[-1,num_hidden*2]), units=num_hidden,activation=tf.nn.relu, trainable=True)
            denses = tf.nn.dropout(denses, self.dropout_keep_prob)
            out_weight = tf.Variable(tf.random_normal([num_hidden, 2]))
            out_bias = tf.Variable(tf.random_normal([2]))
    
            with tf.name_scope("output"):
                self.scores = tf.nn.xw_plus_b(denses, out_weight,out_bias, name="scores")
                self.predictions = tf.nn.softmax(self.scores, name="predictions")
    
            with tf.name_scope("loss"):
                self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.domain)
                self.domain_loss = tf.reduce_mean(self.losses, name="loss")   
                
            with tf.name_scope("accuracy"):
                self.correct_domain_pred = tf.equal(tf.argmax(self.predictions, 1),tf.argmax(self.domain, 1))
                self.domain_acc = tf.reduce_mean(tf.cast(self.correct_domain_pred, "float"),name="accuracy")
                
            self.total_loss = self.pred_loss + self.domain_loss
            
            
        total_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Total number of trainable parameters: %d" % total_parameters)


