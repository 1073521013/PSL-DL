# coding: utf-8
#from cnn_ada import CNN_LSTM  # OPTION 0
from metrics_mc import *

from lstm_cnn import LSTM_CNN   #OPTION 1
istrain = False
# istrain = True
MODEL_TO_RUN = 1
from confusionmatrix import ConfusionMatrix
from sklearn.manifold import TSNE
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from utils import *
import tensorflow as tf
import numpy as np
import os
import time
import h5py
from iterate_minibatches import iterate_minibatches

f = h5py.File('../data/human.hdf5', 'r')
x_human = f["x"]
y_human = f["y"]
indices = np.arange(x_human.shape[0])
#np.random.seed(seed=123456)
np.random.shuffle(indices)
x_human = np.array(x_human)
y_human = np.array(y_human)

x_human = x_human[indices]
y_human = y_human[indices]

# x_human = x_human[:9903]
# y_human = y_human[:9903]
f = h5py.File('../data/thaliana.hdf5', 'r')
x_mouse = f["x"]
y_mouse = f["y"]
indices = np.arange(x_mouse.shape[0])
np.random.shuffle(indices)
x_mouse = np.array(x_mouse)
y_mouse = np.array(y_mouse)
x_mouse = x_mouse[indices]
y_mouse = y_mouse[indices]

test_index = []
train_index = []
for i in range(len(x_human)):
    if i % 4 == 0:
        test_index.append(i)
    else:
        train_index.append(i)
x_human_train = x_human[train_index, :, :]
y_human_train = y_human[train_index, :]
x_human_test = x_human[test_index, :, :]
y_human_test = y_human[test_index, :]

test_indexm = []
train_indexm = []
for i in range(len(x_mouse)):
    if i % 4 == 0:
        test_indexm.append(i)
    else:
        train_indexm.append(i)
x_mouse_train = x_mouse[train_indexm, :, :]
y_mouse_train = y_mouse[train_indexm, :]
x_mouse_test = x_mouse[test_indexm, :, :]
y_mouse_test = y_mouse[test_indexm, :]

print(x_human.shape)
print(x_mouse.shape)

print(x_human_train.shape)
print(y_human_train.shape)

print(x_mouse_train.shape)
print(y_mouse_train.shape)
'''
y=np_utils.to_categorical(y)  
num=int(x.shape[0]/2)
#x_mouse=x[:num,:,:]
#y_mouse=y[:num,:]
#x_human=x[num:,:,:]
#y_human=y[num:,:]
mouse=[]
human = []
for i in range(len(x)):
    if i%2 == 0:
        mouse.append(i)
    else :
        human.append(i)
x_mouse=x[mouse,:,:]
y_mouse=y[mouse,:]
x_human=x[human,:,:]
y_human=y[human,:]


test_index=[]
train_index = []
for i in range(num):
    if i%3 == 0:
        test_index.append(i)
    else :
        train_index.append(i)

indices = np.arange(x_mouse.shape[0])
np.random.seed(seed=123456)
np.random.shuffle(indices)
x_human = x_human[indices]
y_human = y_human[indices]
print(x_human.shape)
print(y_human.shape)
x_human_test=x_human[test_index,:,:]
x_human_train=x_human[train_index,:,:]
y_human_test=y_human[test_index,:]
y_human_train=y_human[train_index,:]

np.random.seed(seed=12345678)
np.random.shuffle(indices)
x_mouse = x_mouse[indices]
y_mouse = y_mouse[indices]
print(x_mouse.shape)
print(y_mouse.shape)
x_mouse_test=x_mouse[test_index,:,:]
x_mouse_train=x_mouse[train_index,:,:]
y_mouse_test=y_mouse[test_index,:]
y_mouse_train=y_mouse[train_index,:]
'''
num_test = 700
a1 = 300
a2 = a1 + num_test
combined_test_imgs = np.vstack([x_mouse_test[a1:a2, :, :], x_human_test[a1:a2, :, :]])
combined_test_labels = np.vstack([y_mouse_test[a1:a2, :], y_mouse_test[a1:a2, :]])
combined_test_domain = np.vstack([np.tile([0., 1.], [num_test, 1]), np.tile([1., 0.], [num_test, 1])])

# Parameters
# Model Hyperparameters
n_class = 10  # 128
seq_legth = 1000
embedding_dim = 21
num_hidden = 80
filter_sizes = [1, 3, 5, 9, 15, 21]  # 3
num_filters = 20
dropout_prob = 0.5  # 0.5
l2_reg_lambda = 0.0
n = 1
# Training parameters

batch_size = 32
num_epochs = 500 * 40  # 400
num_checkpoints = 1  # Checkpoints to store# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # embed()
        if (MODEL_TO_RUN == 0):
            model = CNN_LSTM(seq_legth, n_class, embedding_dim, filter_sizes, num_filters, num_hidden, batch_size)
        elif (MODEL_TO_RUN == 1):
            model = LSTM_CNN(seq_legth, n_class, embedding_dim, filter_sizes, num_filters, num_hidden, batch_size)
        else:
            print("PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n")
            exit();
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # learning_rate = tf.train.exponential_decay(0.002,global_step,100,0.99,staircase=True)
        # learning_rate = tf.placeholder(tf.float32, [])
        optimizer = tf.train.AdamOptimizer()
        regular_train_op = optimizer.apply_gradients(optimizer.compute_gradients(model.pred_loss),
                                                     global_step=global_step)
        grads_and_vars = optimizer.compute_gradients(model.total_loss)
        dann_train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_sou", timestamp))
        print("Writing to {}\n".format(out_dir))
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.total_loss)
        acc_summary = tf.summary.scalar("accuracy", model.label_acc)
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join("runs_sou", "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        # TRAINING STEP
        def train_step(x_batch, y_batch, domain_labels, l, save=True):
            # feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: dropout_prob,
            # model.training: True} _,predictions, step, summaries, loss,x2,x3 = sess.run([regular_train_op,
            # model.scores, global_step, train_summary_op, model.loss,model.val,model.semantic],feed_dict)
            _, step, summaries, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                [dann_train_op, global_step, train_summary_op, model.total_loss, model.domain_loss, model.pred_loss,
                 model.domain_acc, model.label_acc],
                feed_dict={model.input_x: x_batch, model.input_y: y_batch, model.domain: domain_labels,
                           model.training: True, model.dropout_keep_prob: dropout_prob, model.l: l})
            if save:
                train_summary_writer.add_summary(summaries, step)
            return batch_loss, dloss, ploss, d_acc, p_acc


        # EVALUATE MODEL
        def dev_step(x_batch, y_batch, save=True):
            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: 1,
                         model.training: False}
            predictions, step, summaries, loss, x2, x3 = sess.run(
                [model.scores, global_step, dev_summary_op, model.loss, model.val, model.semantic], feed_dict)
            # step,predictions,loss, accuracy = sess.run([global_step,model.scores, model.loss,model.accuracy],
            # feed_dict) time_str = datetime.datetime.now().isoformat() print("{}: step {}, loss {:g},
            # acc {:g}".format(time_str, step, loss, accuracy))
            current_step = tf.train.global_step(sess, global_step)
            if save:
                dev_summary_writer.add_summary(summaries, current_step)
            return loss, predictions, x2, x3


        # CREATE THE BATCHES GENERATOR
        eps = []
        max_acc = 0
        if istrain:
            start_time = time.time()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            # Batch generators
            gen_source_batch = batch_generator([x_mouse_train, y_mouse_train], batch_size // 2)
            gen_target_batch = batch_generator([x_human_train, y_human_train], batch_size // 2)
            gen_source_only_batch = batch_generator([x_mouse_train, y_mouse_train], batch_size)
            gen_target_only_batch = batch_generator([x_human_train, y_human_train], batch_size)
            domain_labels = np.vstack(
                [np.tile([0., 1.], [batch_size // 2, 1]), np.tile([1., 0.], [batch_size // 2, 1])])
            for i in range(num_epochs):
                p = float(i) / num_epochs
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p) ** 0.75
                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)
                #X = np.vstack([X0, X1])
                #y = np.vstack([y0, y1])
                X ,y = next(gen_source_only_batch)
                X ,y = next(gen_target_only_batch)
                batch_loss, dloss, ploss, d_acc, p_acc = train_step(X, y, domain_labels, l, save=True)
                if i % 100 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(i // 460, 20, time.time() - start_time))
                    print('loss: {}  d_acc: {}  p_acc: {}  time: {}  l: {} '.format(batch_loss, d_acc, p_acc,
                                                                                    time.time() - start_time, l))
                    source_acc = sess.run(model.label_acc,
                                          feed_dict={model.input_x: x_mouse_test, model.input_y: y_mouse_test,
                                                     model.dropout_keep_prob: 1, model.training: False})
                    target_acc = sess.run(model.label_acc,
                                          feed_dict={model.input_x: x_human_test, model.input_y: y_human_test,
                                                     model.dropout_keep_prob: 1, model.training: False})
                    d_acc = sess.run(model.domain_acc,
                                     feed_dict={model.input_x: combined_test_imgs, model.domain: combined_test_domain,
                                                model.dropout_keep_prob: 1, model.training: False})
                    if target_acc > max_acc:
                        max_acc = target_acc
                        saver.save(sess, checkpoint_prefix, global_step=tf.train.global_step(sess, global_step))
                        print('\nSaved!!!!!!!!!!Domain adaptation training')
                        print('Source (mouse) accuracy:', source_acc)
                        print('target (human) accuracy:', target_acc)  # Compute final evaluation on test data
                        print('Domain accuracy:', d_acc)
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            source_acc, predictions_ = sess.run([model.label_acc, model.score],
                                               feed_dict={model.input_x: x_mouse_test, model.input_y: y_mouse_test,
                                                          model.dropout_keep_prob: 1, model.training: False})
            target_acc, predictions = sess.run([model.label_acc, model.score],
                                                feed_dict={model.input_x: x_human_test, model.input_y: y_human_test,
                                                           model.dropout_keep_prob: 1, model.training: False})
            d_acc = sess.run(model.domain_acc,
                             feed_dict={model.input_x: combined_test_imgs, model.domain: combined_test_domain,
                                        model.dropout_keep_prob: 1, model.training: False})
            print('\nDomain adaptation training')
            print('Source (mouse) accuracy:', source_acc)
            print('target (human) accuracy:', target_acc)
            print('Domain accuracy:', d_acc)

            outputs = np.argmax(predictions, axis=-1)
            print(outputs.shape)
            confusion_test = ConfusionMatrix(10)
            y_test = np.argmax(y_human_test, axis=-1)
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
            print(a, b, e, f)
            print("FINAL TEST RESULTS")
            print(confusion_test)
            print(cf_val)
            print("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
            print("  test positive_predictive_value:\t\t{:.2f} %".format(positive_predictive_value * 100))
            print("  test negative_predictive_value:\t\t{:.2f} %".format(negative_predictive_value * 100))
            print("  test F1:\t\t{:.2f} %".format(F1 * 100))
            print("  test MCC:\t\t{:.2f} %".format(MCC * 100))
            print("  test kappa:\t\t{:.2f}".format(kappa(cf_val)))

            plt.figure(figsize=(12, 8),dpi=80)
            cmap = plt.cm.Blues
            plt.imshow(cf_val, interpolation='nearest', cmap=cmap)
            plt.title('Confusion matrix validation set')
            plt.colorbar()
            tick_marks = np.arange(10)
            classes = ['Cell.membrane', 'Cytoplasm', 'Endoplasmic', 'Extracellular', "Golgi.apparatus", "Lysosome",
                        "Mitochondrion", "Nucleus", "Peroxisome", "Plastid"]
            plt.xticks(tick_marks, classes, rotation=60)
            plt.yticks(tick_marks, classes)

            thresh = cf_val.max() / 2.
            for i, j in itertools.product(range(cf_val.shape[0]), range(cf_val.shape[1])):
                plt.text(j, i, cf_val[i, j], horizontalalignment="center",
                         color="white" if cf_val[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True location')
            plt.xlabel('Predicted location')
            plt.savefig("test.jpg")

            # assert isinstance(model, object)
            # test_emb = sess.run(model.feature, feed_dict={model.input_x: combined_test_imgs, model.dropout_keep_prob: 1,
            #                                               model.training: False})
            # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
            # source_only_tsne = tsne.fit_transform(test_emb)
            # plot_embedding(source_only_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1),'Domain adaptation')
            ##############################################
            class_names = ['Cell.membrane', 'Cytoplasm', 'Endoplasmic', 'Extracellular', "Golgi.apparatus", "Lysosome",
                           "Mitochondrion", "Nucleus", "Peroxisome", "Plastid"]

            from sklearn.preprocessing import OneHotEncoder
            from sklearn.metrics import roc_curve, auc, roc_auc_score
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
            plt.gca().set_color_cycle(
                ['red', 'green', 'blue', 'peru', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan'])
            fpr = dict()
            tpr = dict()
            roc_auc = np.empty(10 + 2)
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
                co.append(class_name + '(AUC={:.2f})'.format(roc_auc[i]))
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
            # plt.title('ROC of %s ' % class_name + '(AUC={:.2f}, Thr={:.2}, Acc={:.2f}%'.format(roc_auc[class_name], thresholds_val,  acc * 100))
            # plt.legend(co)
            plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
            plt.show()
            # plt.savefig('jpg/roc_%s.jpg'%class_name)
            plt.savefig('all_roc00.jpg')



