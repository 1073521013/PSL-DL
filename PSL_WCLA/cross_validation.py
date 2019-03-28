# -*- coding: utf-8 -*-
#! /usr/bin/env python
#SELECT WHICH MODEL YOU WISH TO RUN:
#from cnn_lstm import CNN_LSTM   #OPTION 0
from lstm_cnn import LSTM_CNN   #OPTION 1
#from cnn import CNN             #OPTION 2 (Model by: Danny Britz)
#from lstm import LSTM           #OPTION 3
MODEL_TO_RUN = 1
#istrain= False
istrain= True
import tensorflow as tf
import numpy as np
import os
import random
import time
import datetime
import batchgen
from confusionmatrix import ConfusionMatrix
from metrics_mc import *

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
num_epochs = 35 #200
evaluate_every = 20 #100
checkpoint_every = 10000 #100
num_checkpoints = 1 #Checkpoints to store

# Data Preparation
import h5py
f=h5py.File('../data/gensim100_00_300.hdf5','r')
x_train=f['x_train']
x_test=f['x_test']
y_train=f['y_train']
y_test=f['y_test']
y_train = np.array(y_train)
x_train = np.array(x_train)
y_dev = np.array(y_test)
x_dev = np.array(x_test)

#indices0 = np.arange(x_test.shape[0])
#np.random.shuffle(indices0)
#x_dev = x_test[indices0]
#y_dev = y_test[indices0]

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]
print(x_train.shape)##31440*500*100
print(y_test.shape)##7890*10
#y_train = y_train[:400,:]
#x_train = x_train[:400,:,:]
#y_dev = y_dev[:100,:]
#x_dev = x_dev[:100,:,:]
#print(x_train.shape)##31440*500*100
#print(y_dev.shape)##7890*10
# partition set
import itertools
x=[]
for i in range(7860):#7860
    li=list(np.r_[1:5])
    random.shuffle(li)
    x.append(li)
partition=np.array(list(itertools.chain.from_iterable(x)))
outputs=[0]*len(x_dev)
# Training
for i in range(1,5):
	# Network compilation
    print("Compilation model {}\n".format(i))	
	# Train and validation sets
    train_index = np.where(partition != i)
    val_index = np.where(partition == i)
    X_tr = x_train[train_index].astype(np.float32)
    X_val = x_train[val_index].astype(np.float32)
    y_tr = y_train[train_index].astype(np.int32)
    y_val = y_train[val_index].astype(np.int32)

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
            optimizer = tf.train.AdamOptimizer()
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
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_11", timestamp))
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
            checkpoint_dir = os.path.abspath(os.path.join('runs_11', "checkpoints"+str(i)))
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
                _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, model.loss, model.accuracy],feed_dict)
                time_str = datetime.datetime.now().isoformat()
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 50 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if save:
                    train_summary_writer.add_summary(summaries, step)

            #EVALUATE MODEL
            def dev_step(x_batch, y_batch, writer=None,save=True):
                feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: dropout_prob}
                step, summaries, loss,output, accuracy = sess.run([global_step, dev_summary_op, model.loss,model.scores, model.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if save:
                    if writer:
                        writer.add_summary(summaries, step)
                return accuracy,output
            #testing MODEL
#            def test_step(x_batch, y_batch, writer=None,save=True):
#                feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: dropout_prob}
#                step, summaries,output, accuracy = sess.run([global_step, test_summary_op,model.scores, model.accuracy], feed_dict)
#                #step,output, accuracy = sess.run([global_step,model.scores, model.accuracy], feed_dict)
#                time_str = datetime.datetime.now().isoformat()
#                print("{}: step {}, acc {:g}".format(time_str, step, accuracy))
#                if save:
#                    if writer:
#                        writer.add_summary(summaries, step)
#                return output            
            #CREATE THE BATCHES GENERATOR
            batches = batchgen.gen_batch(list(zip(X_tr, y_tr)), batch_size, num_epochs)
            batches_val = batchgen.gen_batch(list(zip(X_val, y_val)), batch_size, num_epochs)
            #TRAIN FOR EACH BATCH
            max_acc=0
            if istrain:
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % evaluate_every == 0:
                        print("\nEvaluation:")
                        accuracy,_ = dev_step(X_val, y_val, writer=dev_summary_writer)
                        print("")
                        if accuracy>max_acc:
                            print("\nTest:")
                            _,output = dev_step(x_dev, y_dev, writer=test_summary_writer)
                            max_acc=accuracy
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                outputs = [output[j]+outputs[j] for j in range(len(x_dev))]
            else:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
                saver.restore(sess,ckpt.model_checkpoint_path)     
                output = dev_step(x_dev, y_dev)
                outputs = [output[j]+outputs[j] for j in range(len(x_dev))]
outputs = np.array(outputs) / 4.0
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

