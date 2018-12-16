from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# %% Initialization
X    = tf.placeholder(tf.float32, [None, 28, 28, 1])
W    = tf.Variable(tf.zeros([784, 10]))
b    = tf.Variable(tf.zeros([10]))
init = tf.global_variables_initializer()

# %% Model
Y  = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

# %% Placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

# %% Loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# %% Percentage of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy   = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# %% Training step
optimizer  = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess  = tf.Session()
sess.run(init)

for i in range(1):
    # Load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    batch_X          = batch_X.reshape(-1, 28, 28, 1)
    train_data       = {X: batch_X, Y_: batch_Y}

    # Train
    sess.run(train_step, feed_dict=train_data)

    # If success add code to print it
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    # If success on test data
    test_X    = mnist.test.images.reshape(-1, 28, 28, 1)
    test_Y    = mnist.test.labels
    test_data = {X: test_X, Y_: test_Y}
    a, c      = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    print("batch:{}, acc: {}, loss: {}" % (i, a, c))
