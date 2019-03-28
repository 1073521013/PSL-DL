# -*- coding: utf-8 -*-
#! /usr/bin/env python
#SELECT WHICH MODEL YOU WISH TO RUN:
#from self_cnn import CNN_LSTM   #OPTION 0
from cnn1 import CNN_LSTM   #OPTION 0
from lstm_cnn import LSTM_CNN   #OPTION 1
istrain= False
#istrain= True      
MODEL_TO_RUN = 0
from confusionmatrix import ConfusionMatrix
from metrics_mc import *
import tensorflow as tf
import numpy as np
import os
import time
import h5py

#import batchgen
from iterate_minibatches import iterate_minibatches
# Parameters
# ==================================================
# Model Hyperparameters
n_class  = 10     #128
seq_legth = 1000 
embedding_dim = 21
num_hidden = 128 
filter_sizes = [1,3,5,9,15,21]  #3
num_filters = 20
dropout_prob = 0.5 #0.5
l2_reg_lambda = 0.0

# Training parameters
learning_rate = 0.0005
batch_size = 100
num_epochs = 100 #200
num_checkpoints = 1 #Checkpoints to store

# Data Preparation
f=h5py.File('../data/human.hdf5','r')
x=f["x"]
y=f["y"]
x=np.array(x)
y=np.array(y)
num=x.shape[0]
indices = np.arange(num)
np.random.seed(seed=123456)
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
#y=np_utils.to_categorical(y)  
test_index=[]
train_index = []
for i in range(num):
    if i%4 == 0:
        test_index.append(i)
    else :
        train_index.append(i)

x_dev=x[test_index,:,:]
x_train=x[train_index,:,:]
y_dev=y[test_index]
y_train=y[train_index]


indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]
print(x_train.shape)
print(y_train.shape)
#import h5py
#f=h5py.File('../data/gensim100_00_300.hdf5','r')
#x_train=f['x_train']
#x_test=f['x_test']
#y_train=f['y_train']
#y_test=f['y_test']
#y_train = np.array(y_train)
#x_train = np.array(x_train)
#y_dev = np.array(y_test)
#x_dev = np.array(x_test)
#print(x_dev.shape)
#print(y_dev.shape)
#
##y_train = y_train[:2000,:]
##x_train = x_train[:2000,:,:]
##y_dev = y_dev[:500,:]
##x_dev = x_dev[:500,:,:]
#indices = np.arange(x_train.shape[0])
#np.random.shuffle(indices)
#x_train = x_train[indices]
#y_train = y_train[indices]
#print(x_train.shape)
#print(y_train.shape)
#embed()


# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #embed()
        if (MODEL_TO_RUN == 0):
            model = CNN_LSTM(seq_legth,n_class,embedding_dim,filter_sizes,num_filters,num_hidden)
        elif (MODEL_TO_RUN == 1):
            model = LSTM_CNN(seq_legth,n_class,embedding_dim,filter_sizes,num_filters,num_hidden)
        else:
            print ("PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n")
            exit();
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #learning_rate = tf.train.exponential_decay(0.002,global_step,100,0.99,staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs0_self", timestamp))
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
        checkpoint_dir = os.path.abspath(os.path.join("runs0_self", "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        #TRAINING STEP
        def train_step(x_batch, y_batch,save=True):
            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: dropout_prob,model.training: True}
            _,predictions, step, summaries, loss = sess.run([train_op,model.scores, global_step, train_summary_op, model.loss],feed_dict)
            #time_str = datetime.datetime.now().isoformat()
            #current_step = tf.train.global_step(sess, global_step)
            #if current_step % 50 == 0:
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                train_summary_writer.add_summary(summaries, step)
            return loss ,predictions
        #EVALUATE MODEL
        def dev_step(x_batch, y_batch,save=True):
            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: 1,model.training: False}
            predictions,step,summaries,loss = sess.run([model.scores,global_step, dev_summary_op, model.loss], feed_dict)
            #step,predictions,loss, accuracy = sess.run([global_step,model.scores, model.loss,model.accuracy], feed_dict)            
            #time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            current_step = tf.train.global_step(sess, global_step)
            if save:
                dev_summary_writer.add_summary(summaries, current_step)
            return loss ,predictions
        #CREATE THE BATCHES GENERATOR
        eps = []
        max_acc=0
        if istrain:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            for epoch in range(num_epochs):
               start_time = time.time()
        	     # Full pass training set
               train_err = 0
               train_batches = 0
               confusion_train = ConfusionMatrix(n_class)
    	    
    	        # Generate minibatches and train on each one of them	
               for batch in iterate_minibatches(x_train, y_train, batch_size, shuffle=True):
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
               confusion_valid = ConfusionMatrix(n_class)
    	         # Generate minibatches and train on each one of them	
               for batch in iterate_minibatches(x_dev, y_dev, batch_size):
                   inputs, targets = batch
                   err, predict_val = dev_step(inputs, targets)
                   val_err += err
                   val_batches += 1
                   preds = np.argmax(predict_val, axis=-1)
                   targets = np.argmax(targets, axis=-1)
                   confusion_valid.batch_add(targets, preds)
    		
               val_loss = val_err / val_batches
               val_accuracy = confusion_valid.accuracy()
               cf_val = confusion_valid.ret_mat()               
               if val_accuracy>max_acc:
                   max_acc=val_accuracy
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
            val_err = 0
            val_batches = 0    
            confusion_valid = ConfusionMatrix(n_class)
            for batch in iterate_minibatches(x_dev, np_utils.to_categorical(y_dev), batch_size):
                inputs, targets = batch
                err, predict_val,_,_ = dev_step(inputs, targets,save=False)
                val_err += err
                val_batches += 1
                preds = np.argmax(predict_val, axis=-1)
                targets = np.argmax(targets, axis=-1)
                confusion_valid.batch_add(targets, preds)
    		
            val_loss = val_err / val_batches
            val_accuracy = confusion_valid.accuracy()
            cf_val = confusion_valid.ret_mat()        
            print (confusion_valid)
            print("  validation loss:\t\t{:.6f}".format(val_loss))
            print("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
            print("  validation Gorodkin:\t\t{:.2f}".format(gorodkin(cf_val)))
            print("  validation IC:\t\t{:.2f}".format(IC(cf_val)))              
#            X_test = sess.run([model.val], {model.input_x: x_dev, model.dropout_keep_prob: 1,model.training: False})  
#            X_train = sess.run([model.val], {model.input_x: x_train, model.dropout_keep_prob: 1,model.training: False}) 
#            print(X_train.shape)
#            print(y_train.shape)
#            print(X_test.shape)
#            print(y_dev.shape)
#            import h5py
#            f=h5py.File('data/predict.hdf5')
#            #spec_dtype = h5py.special_dtype(vlen=np.dtype('float32'))
#            d1=f.create_dataset('x_train',data=X_train)
#            d2=f.create_dataset('x_test',data=X_test)
#            d3=f.create_dataset('y_train',data=y_train)
#            d4=f.create_dataset('y_test',data=y_dev)
