"""
    training a deep neural network on the mnist dataset
    - build the neuron layer
    - build the cost function
    - build the loss optimizer
    - build the accuracy calculator
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from datetime import datetime


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)


mnist = fetch_mldata('MNIST original')
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data / 255, mnist.target, random_state=0, test_size=0.2
)


n_epochs = 40
batch_size = 50


n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")



with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")


# defined the cost function

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) # comparing logits against y
    loss = tf.reduce_mean(xentropy, name="loss")


# created an optimizer to reduce the loss

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# defined the performance measure

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# node to initialize all variables and a saver() method

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# ============================== the execution phase ========================================= #


def next_batch(num, data, labels):
    """
    Return a total of 'num' random samples and labels. 
    """
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)



with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in range(len(X_train) // batch_size):
            X_batch, y_batch = next_batch(batch_size, X_train, y_train)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy: ", acc_train, "Test accuracy:", acc_test)

    #save_path = saver.save(sess, "./modified_my_model_final.ckpt")


# ============== to use the model ================ #
"""
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")
        X_new_scaled = [...] # some new images (scaled from 0 to 1)
        Z = logits.eval(feed_dict = {X: X_new_scaled})
        y_pred = np.argmax(Z, axis=1)
"""



"""
    # ============ preparing the tensor board ============== #
    now = datetime.utcnow().strftime("%Y%m%d%H%MS")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    for batch_index in range(n_batches)
"""