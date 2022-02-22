import tensorflow as tf

delta = tf.constant(1.)
x = tf.placeholder(tf.float32, None)

def left(x):
    return tf.multiply(x, x) / 2.
def right(x):
    return tf.multiply(delta, tf.abs(x) - delta / 2.)

hubber = tf.cond(tf.abs(x) <= delta,
                 lambda: left(x),
                 lambda: right(x))

sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    print(sess.run(hubber, feed_dict = {x: 0.5}))
    print(sess.run(hubber, feed_dict = {x: 1.0}))
    print(sess.run(hubber, feed_dict = {x: 2.0}))
    print(sess.run(x, feed_dict={x: [1,2,3,4,5]}))