import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata


n_inputs = 28 * 28
n_hidden1 = 600
n_hidden2 = 300
n_outputs = 10
learning_rate = 0.1
epochs = 5
batch_size = 50

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
y = tf.placeholder(tf.int64, shape=[None])


mnist = fetch_mldata('MNIST Original')
sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, random_state=0)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)



# prepare the convolution

    # reshape the data into something that can be passed through the cnn
data = tf.cast((tf.reshape(X_train_std, shape=[-1, 28, 28, 1])), tf.float32)
conv1 = tf.layers.conv2d(data, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu)
# then pass through a pooling function
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
# then through another convolution
conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=5, padding='SAME', activation=tf.nn.relu)
# then through another pooling function
pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='VALID')
# then get a third convolution
conv3 = tf.layers.conv2d(pool2, filters=16, kernel_size=7, padding='SAME', activation=tf.nn.relu)

# then flatten the last convolution into a fully connected layer for dnn
fcl = tf.contrib.layers.flatten(conv3)



with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(fcl, n_hidden1, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs)


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


with tf.name_scope("evaluate"):
    correct = tf.nn.in_top_k(logits, y_train, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

def next_batch(num, data, labels):
    """
        Return a total of 'num' number of samples
    """
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


with tf.Session() as sess:
    init.run()
    for n_epochs in range(epochs):
        for iteration in range(len(X_train // batch_size)):
            X_batch, y_batch = next_batch(batch_size, X_train_std, y_train)
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print(epoch, "Train accuracy: ", acc_train)
    saver_path = saver.save(sess, './testconv.py')