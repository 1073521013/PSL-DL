# -*- coding: utf-8 -*-
# ! /usr/bin/env python
# SELECT WHICH MODEL YOU WISH TO RUN:
from lstm_cnn_new import LSTM_CNN  # OPTION 1
from cnn_lstm import CNN_LSTM  # OPTION 1
#istrain= False
istrain = True
MODEL_TO_RUN = 1
from confusionmatrix import ConfusionMatrix
from metrics_mc import *
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import batchgen
#from keras.utils import np_utils
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools

# Parameters
# ==================================================
# Model Hyperparameters
num_classes = 10  # 128
seq_legth = 300
embedding_dim = 200 ######################################
num_hidden = 100   ######################################
filter_sizes = [1,3,5,9,15,21]
num_filters = 20
dropout_prob = 0.7 # 0.5
learning_rate = 0.0005
l2_reg_lambda = 0.0

# Training parameters
batch_size = 50
num_epochs =  60 # 60
evaluate_every = 20  # 100
checkpoint_every = 10000  # 100
num_checkpoints = 1  # Checkpoints to store

# Data Preparation
import h5py

f = h5py.File('../data/new_glove_200.hdf5', 'r') ##############################
x = f['x']
y = f['y']
x = np.array(x)
y = np.array(y)

num = x.shape[0]
indices = np.arange(num)
np.random.seed(seed=123456)
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

test_index = []
train_index = []
for i in range(num):
    if i % 4 == 0:
        test_index.append(i)
    else:
        train_index.append(i)

x_test = x[test_index, :, :]
x_train = x[train_index, :, :]
y_test = y[test_index, :]
y_train = y[train_index, :]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# embed()

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # embed()
        if (MODEL_TO_RUN == 0):
            model = CNN_LSTM(seq_legth, num_classes, embedding_dim, filter_sizes, num_filters, num_hidden)
        elif (MODEL_TO_RUN == 1):
            model = LSTM_CNN(seq_legth, num_classes, embedding_dim, filter_sizes, num_filters, num_hidden)
        #        elif (MODEL_TO_RUN == 2):
        #            model = CNN(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),
        #                        embedding_dim,filter_sizes,num_filters,l2_reg_lambda)
        #        elif (MODEL_TO_RUN == 3):
        #            model = LSTM(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),embedding_dim)
        else:
            print("PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n")
            exit();

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #learning_rate = tf.train.exponential_decay(0.002,global_step,100,0.96,staircase=True)
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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "f_cbow240", timestamp))
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
        checkpoint_dir = os.path.abspath(os.path.join("f_cbow240", "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        # TRAINING STEP
        def train_step(x_batch, y_batch, save=True):
            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: dropout_prob}
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 50 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                train_summary_writer.add_summary(summaries, step)


        # EVALUATE MODEL
        def dev_step(x_batch, y_batch, writer=None, save=True):
            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: 1}
            step, predictions, summaries, loss, accuracy = sess.run(
                [global_step, model.scores, dev_summary_op, model.loss, model.accuracy], feed_dict)
            # step,predictions,loss, accuracy = sess.run([global_step,model.scores, model.loss,model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                if writer:
                    writer.add_summary(summaries, step)
            return accuracy, predictions


        # CREATE THE BATCHES GENERATOR
        batches = batchgen.gen_batch(list(zip(x_train, y_train)), batch_size, num_epochs)
        # TRAIN FOR EACH BATCH
        max_acc = 0
        if istrain:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            for i, batch in enumerate(batches):
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    accuracy, predictions = dev_step(x_test, y_test, writer=dev_summary_writer)
                    if accuracy > max_acc:
                        max_acc = accuracy
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(x_test.shape)
            print(y_test.shape)
            accuracy, predictions = dev_step(x_test, y_test)  # dev_step(x_dev, y_dev, writer=dev_summary_writer)
outputs = np.argmax(predictions, axis=-1)
print(outputs.shape)
confusion_test = ConfusionMatrix(num_classes)
y_test = np.argmax(y_test, axis=-1)
print(y_test.shape)
confusion_test.batch_add(y_test, outputs)
test_accuracy = confusion_test.accuracy()

positive_predictive_value = confusion_test.positive_predictive_value()
negative_predictive_value = confusion_test.negative_predictive_value()
F1 = confusion_test.F1()
MCC = confusion_test.matthews_correlation()

cf_val = confusion_test.ret_mat()

print("FINAL TEST RESULTS")
print(confusion_test)
print("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))

a, positive_predictive_value = confusion_test.positive_predictive_value()
b, negative_predictive_value = confusion_test.negative_predictive_value()
e, F1 = confusion_test.F1()
f, MCC = confusion_test.matthews_correlation()

print("FINAL TEST RESULTS")
print(confusion_test)
print(cf_val)
print("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
print(a)
print(b)
print(e)
print(f)
print("  test positive_predictive_value:\t\t{:.2f} %".format(positive_predictive_value * 100))
print("  test negative_predictive_value:\t\t{:.2f} %".format(negative_predictive_value * 100))
print("  test F1:\t\t{:.2f} %".format(F1 * 100))
print("  test MCC:\t\t{:.2f} %".format(MCC * 100))
print("  test kappa:\t\t{:.2f}".format(kappa(cf_val)))
#plt.figure(figsize=(8,7),dpi=80)
plt.figure(figsize=(12, 8),dpi=80)
cmap = plt.cm.Blues
plt.imshow(cf_val, interpolation='nearest', cmap=cmap)
plt.title('Confusion matrix validation set')
plt.colorbar()
tick_marks = np.arange(10)
classes = ['Cell.membrane', 'Cytoplasm', 'Endoplasmic', 'Extracellular', "Golgi.apparatus",
           "Lysosome", "Mitochondrion", "Nucleus", "Peroxisome", "Plastid"]
plt.xticks(tick_marks, classes, rotation=60)
plt.yticks(tick_marks, classes)

thresh = cf_val.max() / 2.
for i, j in itertools.product(range(cf_val.shape[0]), range(cf_val.shape[1])):
    plt.text(j, i, cf_val[i, j], horizontalalignment="center", color="white" if cf_val[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True location')
plt.xlabel('Predicted location')
plt.savefig("save/test01.jpg")

##############################################
class_names=['Cell.membrane', 'Cytoplasm', 'Endoplasmic', 'Extracellular', "Golgi.apparatus",
           "Lysosome", "Mitochondrion", "Nucleus", "Peroxisome", "Plastid"]

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc ,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from numpy import interp
enc1 = LabelEncoder()
y_test = enc1.fit_transform(y_test)
enc = OneHotEncoder(sparse=False)
enc.fit(y_test.reshape(-1, 1))
onehot = enc.transform(y_test.reshape(-1, 1))
print(onehot.shape)
print(y_test.shape)
print(predictions.shape)
fig2 = plt.figure('fig2')
plt.gca().set_color_cycle(['red', 'green', 'blue', 'peru','orange','purple','brown','pink','gray','cyan'])
fpr = dict()
tpr = dict()
roc_auc = np.empty(10+2)
thresholds = dict()
co = []
for (i, class_name) in enumerate(class_names):
    fpr[class_name], tpr[class_name], thresholds[class_name] = roc_curve(onehot[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[class_name], tpr[class_name])

    youdens = tpr[class_name] - fpr[class_name]
    index = np.argmax(youdens)
    # youden[class_name]=tpr[class_name](index)
    fpr_val = fpr[class_name][index]
    tpr_val = tpr[class_name][index]
    thresholds_val = thresholds[class_name][index]

    p_auto = predictions[:, i].copy()
    t_auto = onehot[:, i].copy()
    p_auto[p_auto >= thresholds_val] = 1
    p_auto[p_auto < thresholds_val] = 0
    acc = np.float(np.sum(t_auto == p_auto)) / t_auto.size
    co.append(class_name+'(AUC={:.2f})'.format(roc_auc[i]))
    plt.plot(fpr[class_name], tpr[class_name], lw=2, label=class_name + '(%0.2f)' % roc_auc[i])

# micro
fpr['micro'], tpr['micro'], thresholds = roc_curve(onehot.ravel(), predictions.ravel())
roc_auc[10] = auc(fpr['micro'], tpr['micro'])
#plt.plot(fpr['micro'], tpr['micro'], c='r', lw=2, ls='-', alpha=0.8, label=u'micro,AUC=%.3f' % roc_auc[10])

# macro
fpr['macro'] = np.unique(np.concatenate([fpr[i] for i in class_names]))
tpr_ = np.zeros_like(fpr['macro'])
for i in class_names:
    tpr_ += interp(fpr['macro'], fpr[i], tpr[i])
tpr_ /= 10
tpr['macro'] = tpr_
roc_auc[10 + 1] = auc(fpr['macro'], tpr['macro'])
print(roc_auc)
print('Macro AUC:', roc_auc_score(onehot, predictions, average='macro'))
print('Micro AUC:', roc_auc_score(onehot, predictions, average='micro'))
#plt.plot(fpr['macro'], tpr['macro'], c='m', lw=2, alpha=0.8, label=u'macro,AUC=%.3f' % roc_auc[10+1])

plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC of each class')
#plt.title('ROC of %s ' % class_name + '(AUC={:.2f}, Thr={:.2}, Acc={:.2f}%'.format(roc_auc[class_name], thresholds_val,  acc * 100))
# plt.legend(co)
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.show()
#plt.savefig('jpg/roc_%s.jpg'%class_name)
plt.savefig('all_roc00.jpg')

