import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

T, F = 1., -1.

train_in = [[T, T], [T, F], [F, T], [F, F]]
train_out = [[F], [T], [T], [F]]

w1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.zeros([2]))
out1 = tf.tanh(tf.add(tf.matmul(train_in, w1), b1))

w2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.zeros([1]))
out2 = tf.tanh(tf.add(tf.matmul(out1, w2), b2))

error = tf.subtract(train_out, out2)
mse = tf.reduce_mean(tf.square(error))

train = tf.train.GradientDescentOptimizer(0.05).minimize(mse)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
err, target = 1, 0.01
epoch, max_epochs = 0, 2000

def test():
    print('\nweight 1\n' % sess.run(w1))
    print('bias 1\n', sess.run(b1))
    print('weight 2\n', sess.run(w2))
    print('bias 2\n', sess.run(b2))
    print('output\n', sess.run(out2))
    print('mse: ', sess.run(mse), '\n')

# B11: main session
test()
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch:', epoch, 'mse:', err)
test()