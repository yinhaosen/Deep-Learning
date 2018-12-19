from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np

# %% Initialization
X    = tf.placeholder(tf.float32, [None, 28, 28, 1])
W    = tf.Variable(tf.zeros([784, 10]))
b    = tf.Variable(tf.zeros([10]))
init = tf.global_variables_initializer()

# %% Softmax model
Y  = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

# %% Placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

# %% Loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# %% Training step with the learning rate 0.01
optimizer  = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cross_entropy)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess  = tf.Session()
sess.run(init)

for i in range(10000):
    # Load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    batch_X          = batch_X.reshape(-1, 28, 28, 1)
    train_data       = {X: batch_X, Y_: batch_Y}

    # Train data
    sess.run(train_step, feed_dict=train_data)

    # If success on test data
    test_X      = mnist.test.images.reshape(-1, 28, 28, 1)
    test_Y      = mnist.test.labels
    test_data   = {X: test_X, Y_: test_Y}

    # Percentage of correct answers found in batch called accuracy
    # Percentage of correctly predicted positive observations to the total predicted positive observations called precision
    # Percentage of correctly predicted positive observations to the all observations in actual class called recall
    # F1 Score is the weighted average of precision and recall
    predicted_Y = sess.run(Y, feed_dict=test_data)
    predict     = np.argmax(predicted_Y, 1)
    truth       = np.argmax(test_Y, 1)
    a           = accuracy_score(truth, predict)
    p           = precision_score(truth, predict, average=None)
    r           = recall_score(truth, predict, average=None)
    f           = f1_score(truth, predict, average=None)
    c           = sess.run(cross_entropy, feed_dict=test_data)
    print("batch: {}, acc: {}, pre: {}, re: {}, f1: {}, loss: {}".format(i, a, p, r, f, c))
