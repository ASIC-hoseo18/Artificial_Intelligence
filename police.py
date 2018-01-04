# linear.py: linear regression 
# written by Sung Kyu Lim
# limsk@ece.gatech.edu
# 12/15/2017

# B1: import tensorflow
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# B2: variables used
# we want uniform distribution with 
# mean=0, stddev=1, range=[-1, 1]
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# B3: placeholders used
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# B4: linear model
crime = W * x + b

# B5: loss function using mean square error
cost = tf.reduce_mean(tf.square(crime - y))

# B6: how we update weight and bias0000000
opt = tf.train.GradientDescentOptimizer(learning_rate=0.005)

# B7: objective during training
train = opt.minimize(cost)

# B8: session definition
train_tot = 10000
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# B9: training data
x_tr = [10, 15, 16, 1, 4, 6, 18, 12, 14, 7]
y_tr = [5.0, 2.0, 1.0, 9.0, 7.0, 8.0, 1.0, 5.0, 3.0, 6.0]

# B10: do training
for i in range(train_tot):
    error, _ = sess.run([cost, train], feed_dict={x: x_tr, y: y_tr})
    print(i, 'error = %.3f' % error, 'W = %.3f' % sess.run(W), 'b = %.3f' % sess.run(b))

# B11: model testing
test = 30
guess = sess.run(crime, feed_dict={x: test})
print('\nofficers =', test, 'crime = %.3f' % guess)