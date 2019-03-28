# -*- coding: utf-8 -*-
#SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM   #OPTION 0
from lstm_cnn import LSTM_CNN   #OPTION 1
MODEL_TO_RUN = 1
istrain= False
#istrain= True
import tensorflow as tf
import numpy as np
import os
import random
import time
import datetime
from iterate_minibatches import iterate_minibatches
from confusionmatrix import ConfusionMatrix
from metrics_mc import *
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
# Parameters
# ==================================================
# Model Hyperparameters
num_classes  = 10     #128
seq_legth = 300 
embedding_dim = 100
num_hidden = 50
filter_sizes = [1,5,9]  #3
num_filters = 50
dropout_prob = 0.5 #0.5
l2_reg_lambda = 0.0

# Training parameters
batch_size = 100
num_epochs = 30 #200
num_checkpoints = 1 #Checkpoints to store

# Data Preparation
import h5py
f=h5py.File('../data/gensim100_1.hdf5','r')
x_train1=f['x_train1']
x_test1=f['x_test1']
x_train2=f['x_train2']
x_test2=f['x_test2']
x_train3=f['x_train3']
x_test3=f['x_test3']
#y_train=f['y_train']
#y_test=f['y_test']
Y_train=f['Y_train']
test_y = pd.read_csv('../data/test_y.csv',header =None)
le = LabelEncoder()
Y_test = np_utils.to_categorical(le.fit_transform(test_y))

x_test = np.concatenate((x_test1,x_test2,x_test3))
y_test = np.concatenate((Y_test,Y_test,Y_test))

y_dev = np.array(Y_test)
x_dev = np.array(x_test1)
print(x_dev.shape)
print(y_dev.shape)

x_train1=np.array(x_train1)
x_train2=np.array(x_train2)
x_train3=np.array(x_train3)
Y_train=np.array(Y_train)

indices = np.arange(len(Y_train))
np.random.shuffle(indices)
x_train1 = x_train1[indices]
y_train1 = Y_train[indices]
x_train2 = x_train2[indices]
y_train2 = Y_train[indices]
x_train3 = x_train3[indices]
y_train3 = Y_train[indices]

print(x_train1.shape)##31440*500*100 /3
print(y_train1.shape)##7890*10 /3
#y_train = y_train[:400,:]
#x_train = x_train[:400,:,:]
#y_dev = y_dev[:100,:]
#x_dev = x_dev[:100,:,:]
#print(x_train.shape)##31440*500*100
#print(y_dev.shape)##7890*10
# partition set
import itertools
x=[]
for i in range(2620):#7860
    li=list(np.r_[1:5])
    random.shuffle(li)
    x.append(li)
partition=np.array(list(itertools.chain.from_iterable(x)))
outputs=[0]*len(x_dev)
complete_test = np.zeros((x_dev.shape[0],num_classes))
# Training
for i in range(1,5):
	# Network compilation
    print("Compilation model {}\n".format(i))	
	# Train and validation sets
    train_index = np.where(partition != i)
    val_index = np.where(partition == i)
    X_tr1 = x_train1[train_index].astype(np.float32)
    X_val1 = x_train1[val_index].astype(np.float32)
    X_tr2 = x_train2[train_index].astype(np.float32)
    X_val2 = x_train2[val_index].astype(np.float32)
    X_tr3 = x_train3[train_index].astype(np.float32)
    X_val3 = x_train3[val_index].astype(np.float32)
    X_tr = np.concatenate((X_tr1,X_tr2,X_tr3))
    X_val = np.concatenate((X_val1,X_val2,X_val3))
    
    y_tr1 = y_train1[train_index].astype(np.float32)
    y_val1 = y_train1[val_index].astype(np.float32)
    y_tr2 = y_train2[train_index].astype(np.float32)
    y_val2 = y_train2[val_index].astype(np.float32)
    y_tr3 = y_train3[train_index].astype(np.float32)
    y_val3 = y_train3[val_index].astype(np.float32)      
    y_tr = np.concatenate((y_tr1,y_tr2,y_tr3))
    y_val = np.concatenate((y_val1,y_val2,y_val3))

    print("Validation shape: {}".format(X_val.shape))
    print("Training shape: {}".format(X_tr.shape))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            #embed()
            if (MODEL_TO_RUN == 0):
                model = CNN_LSTM(seq_legth,num_classes,embedding_dim,filter_sizes,num_filters)
            elif (MODEL_TO_RUN == 1):
                model = LSTM_CNN(seq_legth,num_classes,embedding_dim,filter_sizes,num_filters,num_hidden)
        #        elif (MODEL_TO_RUN == 2):
        #            model = CNN(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),
        #                        embedding_dim,filter_sizes,num_filters,l2_reg_lambda)
        #        elif (MODEL_TO_RUN == 3):
        #            model = LSTM(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),embedding_dim)
            else:
                print ("PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n")
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
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_1", timestamp))
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
            # test summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)            
            
            
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join('runs_1', "checkpoints"+str(i)))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
            #saver = tf.train.import_meta_graph("runs/1522033132/checkpoints2/model.ckpt-100.meta")
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
        
            #TRAINING STEP
            def train_step(x_batch, y_batch,save=True):
                feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: dropout_prob}
                _,predictions, step, summaries, loss = sess.run([train_op,model.scores, global_step, train_summary_op, model.loss],feed_dict)
                if save:
                    train_summary_writer.add_summary(summaries, step)
                return loss ,predictions
            #EVALUATE MODEL
            def dev_step(x_batch, y_batch,writer=None,save=True):
                feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: 1}
                step, predictions,summaries, loss = sess.run([global_step,model.scores, dev_summary_op, model.loss], feed_dict)
                if save:
                    if writer:
                        writer.add_summary(summaries, step)
                return loss ,predictions

            eps = []
            if istrain:
                for epoch in range(num_epochs):
                   start_time = time.time()
        	    
            	    # Full pass training set
                   train_err = 0
                   train_batches = 0
                   confusion_train = ConfusionMatrix(num_classes)
        	    
        	        # Generate minibatches and train on each one of them	
                   for batch in iterate_minibatches(X_tr, y_tr, batch_size, shuffle=True):
                       inputs, targets = batch
                       tr_err, predict = train_step(inputs, targets)
                       train_err += tr_err
                       train_batches += 1
                       preds = np.argmax(predict, axis=-1)
                       targets = np.argmax(targets, axis=-1)
                       confusion_train.batch_add(targets, preds)
                    	    
                   train_loss = train_err / train_batches
                   train_accuracy = confusion_train.accuracy()
                   cf_train = confusion_train.ret_mat()	   
                   
                   # Full pass validation set
                   val_err = 0
                   val_batches = 0
                   confusion_valid = ConfusionMatrix(num_classes)
        	    
        	        # Generate minibatches and train on each one of them	
                   for batch in iterate_minibatches(X_val, y_val, batch_size):
                       inputs, targets = batch
                       err, predict_val = dev_step(inputs, targets,writer=dev_summary_writer)
                       val_err += err
                       val_batches += 1
                       preds = np.argmax(predict_val, axis=-1)
                       targets = np.argmax(targets, axis=-1)
                       confusion_valid.batch_add(targets, preds)
        		
                   val_loss = val_err / val_batches
                   val_accuracy = confusion_valid.accuracy()
                   cf_val = confusion_valid.ret_mat()               
                   
                   saver.save(sess, checkpoint_prefix, global_step=tf.train.global_step(sess, global_step))
                   eps += [epoch]
                   print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
                   print (confusion_valid)
                   print("  training loss:\t\t{:.6f}".format(train_loss))
                   print("  validation loss:\t\t{:.6f}".format(val_loss))
                   print("  training accuracy:\t\t{:.2f} %".format(train_accuracy * 100))
                   print("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
                   print("  training Gorodkin:\t\t{:.2f}".format(gorodkin(cf_train)))
                   print("  validation Gorodkin:\t\t{:.2f}".format(gorodkin(cf_val)))
                   print("  training IC:\t\t{:.2f}".format(IC(cf_train)))
                   print("  validation IC:\t\t{:.2f}".format(IC(cf_val)))
            else:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
                saver.restore(sess,ckpt.model_checkpoint_path)    
	           # Full pass test set if validation accuracy is higher
        test_batches = 0
        val_batches = 0
        test_pred = np.array([], dtype=np.float32).reshape(0,num_classes)
        for batch in iterate_minibatches(x_dev, y_dev, batch_size, shuffle=False):
            inputs, targets = batch
            err, net_out = dev_step(inputs, targets,writer=test_summary_writer)
            test_batches += 1	
            val_batches += 1
            test_pred = np.concatenate((test_pred, net_out),axis=0)		        
        complete_test += test_pred[:x_dev.shape[0]]
  
test_softmax = complete_test / 4.0
outputs = np.array(test_softmax)
print(outputs.shape)
outputs = np.argmax(outputs,axis=-1)
confusion_test = ConfusionMatrix(num_classes)
y_test = np.argmax(y_dev,axis=-1)
confusion_test.batch_add(y_test, outputs)
test_accuracy = confusion_test.accuracy()
cf_test = confusion_test.ret_mat()


print ("FINAL TEST RESULTS")
print (confusion_test)
print("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
print("  test Gorodkin:\t\t{:.2f}".format(gorodkin(cf_test)))
print("  test IC:\t\t{:.2f}".format(IC(cf_test)))


