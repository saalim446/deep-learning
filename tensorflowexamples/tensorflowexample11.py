import random

import tensorflow as tf

random.seed(1)
a = [random.randint(1,100) for i in range(100)]

a = tf.Variable(a)
b = tf.Variable(a)

mod = tf.mod(a, 2)
equal1 = tf.equal(mod, 1)
equal2 = tf.equal(mod, 0)

oddstarget = tf.fill(a.shape, tf.constant(1), name='oddstarget')
evenstarget = tf.fill(a.shape, tf.constant(0), name='evenstarget')

assign1 = tf.assign(b, tf.where(equal1, oddstarget, a))
assign2 = tf.assign(b, tf.where(equal2, evenstarget, a))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("assign1 ", sess.run(assign1))
    print("b ", b.eval())
    print("assign2 ", sess.run(assign2))
    print("b ", b.eval())
