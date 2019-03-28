#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.decomposition import PCA
from tensorflow.python.framework import ops



def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    idx = data[0].shape[0]
    p = np.random.permutation(idx)
    return [d[p] for d in data]

def batch_gen(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))


def bias_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))


def dense(X, input_dim, units): # phase=True (train)
    W = weight_variable( [input_dim, units] )
    b = bias_variable([units])
    output = tf.matmul(X, W) + b
    return output



def make_meshgrid(x, y, h=.1):
    """Create a mesh of points to plot in
    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(sess, ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = model.predict(sess, np.c_[xx.ravel(), yy.ravel()])
    if Z.shape[1] == 2:
        Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    plt.colorbar(out)
    return out



class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    
flip_gradient = FlipGradientBuilder()




class DANNModel(object):

    def __init__(self, n_features=50, batch_size=32):
        self.n_features = n_features
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self):

        nfeat = 15
        ndom = 15
        nclf = 15

        # self.X : raw input features
        self.X = tf.placeholder(tf.float32, [None, self.n_features])
        # self.y : class labels in [0,1]
        self.y = tf.placeholder(tf.float32, [None, 1])
        # self.domain : domain labels in [0,1]
        self.domain = tf.placeholder(tf.float32, [None, 1])

        # lambda parameter and 'train' flag...
        self.l = tf.placeholder(tf.float32, [], name='l')
        self.train = tf.placeholder(tf.bool, [])

        # first we build a feature extractor
        full_feat = dense(self.X, self.n_features, nfeat)
        full_feat = tf.nn.relu(full_feat)

        #full_feat = dense(full_feat, 50, 50)    
        #full_feat = tf.nn.elu(full_feat, name='full_feat')

        # if we are training here we use only the first half of the
        # the batch (corresponding to labelled source data)..
        select_feat = tf.cond(self.train,
            lambda: tf.slice(full_feat, [0, 0], [self.batch_size/2, -1]),
            lambda: full_feat)

        select_y = tf.cond(self.train,
            lambda: tf.slice(self.y, [0, 0], [self.batch_size/2, -1]),
            lambda: self.y)

        # build the y-classification layer,
        # and set-up the loss for it...
        y_hid = dense(select_feat, nfeat, nclf)
        y_hid = tf.nn.relu(y_hid)

        y_logits = dense(y_hid, nclf, 1)
        self.y_probs = tf.nn.sigmoid(y_logits)

        self.y_crossentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=select_y, 
                logits=y_logits)
            )

        # build the domain-classifier,
        # and set-up the loss for it...
        grl = flip_gradient(full_feat, self.l)

        domain_hid = dense(grl, nfeat, ndom)
        domain_hid = tf.nn.relu(domain_hid)

        domain_hid = dense(domain_hid, ndom, ndom)
        domain_hid = tf.nn.relu(domain_hid)
        
        domain_logits = dense(domain_hid, ndom, 1)
        self.domain_probs = tf.nn.sigmoid(domain_logits)

        self.domain_crossentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.domain,
                logits=domain_logits)
            )

        # during training the DANN minimizes the total loss
        self.total_crossentropy = tf.add(self.y_crossentropy, self.domain_crossentropy)
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.total_crossentropy)

        # this op can be used to train as a standard y-classifier without the domain-adaption.
        self.ytrain_step = tf.train.AdamOptimizer(1e-3).minimize(self.y_crossentropy)

        tf.summary.scalar('tr_total_loss', self.total_crossentropy)
        tf.summary.scalar('tr_domain_loss', self.domain_crossentropy)
        tf.summary.scalar('tr_y_loss', self.y_crossentropy)

        self.merged_summary = tf.summary.merge_all()

    def predict(self, sess, X):
        y_probs = sess.run(
            self.y_probs,
            feed_dict={self.X: X, self.train: False, model.l: 1.0}
            )
        return y_probs # shape=(X.shape[0], 1)



if __name__ == '__main__':

    batch_size = 32

    Xs, ys = make_blobs(300, centers=[[0, 0], [0, 1]], cluster_std=0.2)
    Xt, yt = make_blobs(300, centers=[[1, -1], [1, 0]], cluster_std=0.2)
    Xall = np.vstack([Xs, Xt])
    yall = np.hstack( [ys, yt])
#    plt.scatter(Xall[:, 0], Xall[:, 1], c=yall)
#    plt.savefig('blobs.png')
#    plt.close()

    model = DANNModel(n_features=2)
    S_batches = batch_gen([Xs, ys], batch_size/2)
    T_batches = batch_gen([Xt, yt], batch_size/2)
    
    # 'with' means that the session is destroyed at the end of run
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('tboard/run2')
        writer.add_graph(sess.graph)
    
        for i in range(10000):
    
            p = i/10000.
            lp = 2. / (1. + np.exp(-10*p)) - 1

            X_source, y_source = S_batches.next()
            X_target, y_target = T_batches.next()
            Xbatch = np.vstack( [X_source, X_target] )
            ybatch = np.hstack( [y_source, y_target] )
    
            # first half of batch is from the source batch (=0)
            # second half is the target batch (=1) 
            Dbatch = np.hstack(
                [np.zeros(batch_size/2, dtype=np.int32),
                 np.ones(batch_size/2, dtype=np.int32)]
                )
    
            # resize labels to fit due to sigmoid output
            ybatch = np.reshape(ybatch, (-1, 1))
            Dbatch = np.reshape(Dbatch, (-1, 1))
    
            summary = sess.run(
                model.merged_summary,
                feed_dict={model.X: Xbatch, model.y: ybatch, model.domain: Dbatch,
                           model.train: True, model.l: lp}
                )
            writer.add_summary(summary, i)
    
    
            _, tr_total_loss, tr_domain_loss, tr_class_loss  = sess.run(
                [model.train_step, model.total_crossentropy,
                 model.domain_crossentropy, model.y_crossentropy],
                feed_dict={model.X: Xbatch, model.y: ybatch, model.domain: Dbatch,
                           model.train: True, model.l: lp})
    
            print (i, tr_total_loss, tr_domain_loss, tr_class_loss, lp)

            # after each iteration plot the decision boundary for source and target data
            # for use with ffmpeg for creating decision boundary learning animation...
#            fig, ax = plt.subplots()
#            X0, X1 = Xall[:, 0], Xall[:, 1]
#            xx, yy = make_meshgrid(X0, X1)
#            plot_contours(sess, ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
#            ax.scatter(X0, X1, c=yall)
#            plt.savefig('decision_boundary_{}.png'.format(i))
#            plt.close()
    

        tr_domain = np.zeros(Xs.shape[0], dtype=np.int32)
        tr_domain = np.reshape(tr_domain, (-1, 1))
        te_domain = np.ones(Xt.shape[0], dtype=np.int32)
        te_domain = np.reshape(tr_domain, (-1, 1))

        tr_domain_loss, tr_class_loss  = sess.run(
            [model.domain_crossentropy, model.y_crossentropy],
            feed_dict={model.X: np.reshape(Xs, (-1, 2)), model.y: np.reshape(ys, (-1, 1)), 
                       model.domain: tr_domain,
                       model.train: False, model.l: 1.0})

        te_domain_loss, te_class_loss  = sess.run(
            [model.domain_crossentropy, model.y_crossentropy],
            feed_dict={model.X: np.reshape(Xt, (-1, 2)), model.y: np.reshape(yt, (-1, 1)), 
                       model.domain: te_domain,
                       model.train: False, model.l: 1.0})

        # plot the final decision boundary
#        Xall = np.vstack([Xs, Xt])
#        yall = np.hstack( [ys, yt])
#        fig, ax = plt.subplots()
#        X0, X1 = Xall[:, 0], Xall[:, 1]
#        xx, yy = make_meshgrid(X0, X1)
#        plot_contours(sess, ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
#        ax.scatter(X0, X1, c=yall)
#        plt.savefig('final_decision_boundary.png')
#        plt.close()

        print (tr_domain_loss, tr_class_loss)
        print (te_domain_loss, te_class_loss)
