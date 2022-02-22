import random

import tensorflow as tf

a = tf.Variable([1, 2, 3, 1])
print(a.shape)
print(a.dtype)
cond = tf.equal(a, tf.constant(1))
cond2 = tf.equal(tf.mod(a, 2), 1)
assign1 = tf.assign(a, tf.where(cond, tf.zeros_like(a), a))
assign2 = tf.assign(a, tf.where(cond2, tf.zeros_like(a), a))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(a.eval())
    # a[a==1] = 0
    sess.run(assign1)
    print(a.eval())
    sess.run(assign2)
    print(a.eval())

random.seed(1)
a = [random.randint(1, 100) for i in range(1, 100)]
a = tf.Variable(a)
cond2 = tf.equal(tf.mod(a, 2), 1)
assign2 = tf.assign(a, tf.where(cond2, tf.ones_like(a), a))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(assign2)
    print(a.eval())
