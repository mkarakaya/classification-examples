import tensorflow as tf
from clsdatareader import get_data
import numpy as np


train_X, train_Y, test_X, test_Y = get_data()
N = train_X.shape[0]
D = train_X.shape[1]


def tf_train(X_train, y_train, batch_size=20, n_epoch=1000):
    x = tf.placeholder(tf.float32, [None, D])
    y_ = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.random_normal([D, 1], stddev=1 / np.sqrt(D)))

    # Define loss and optimizer
    z = tf.matmul(x, W)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    # Train
    for epoch in range(n_epoch):
        idx = np.random.choice(len(X_train), batch_size, replace=False)
        _, l = sess.run([train_step, cross_entropy], feed_dict={x: X_train[idx], y_: y_train[idx]})
        if epoch % 100 == 0:
            print('loss: ' + str(l))

    return sess.run(W)


def sigmoid(z):
    return 1/(1+np.exp(-z))


w_est = tf_train(train_X, train_Y * 0.1, batch_size= N)

y_inferred = sigmoid(test_X.dot(w_est)) # Get a probability measure given X
hit = 0
for idx, test in enumerate(test_Y):
    if int(test) == int(y_inferred[idx] * 10):
        hit += 1
print(hit/len(test_Y), hit, len(test_Y))
