import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


df = pd.read_csv('/home/exceptions/datasets/pulsar_stars.csv')

data = df.values
target = df['target_class'].values
data = np.delete(data, -1, axis=1)

sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0, test_size=0.3)

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# ========== the construction phase ============ #
n_epochs = 30
batch_size = 50


n_inputs = 8
n_hidden1 = 100
n_hidden2 = 50
n_outputs = 2
learning_rate = 0.01



X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="X")


with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="logits")


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) # comparing logits against y
    loss = tf.reduce_mean(xentropy, name="loss")


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

# ================== the execution phase =================== #

def next_batch(num, data, labels):
    """
        Function will be used to train the dataset in batches
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
        for i in range(len(X_train_std) // batch_size):
            X_batch, y_batch = next_batch(batch_size, X_train_std, y_train)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test_std, y: y_test})
        print(epoch, "Training accuracy: ", acc_train, " Test accuracy: ", acc_test)
