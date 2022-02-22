import tensorflow as tf

x = tf.placeholder("float",None)
y = x * 2

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x: [1,2,3,4,5]}))

x = tf.placeholder("float",None)
y = x ** 2

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x: [1,2,3,4,5]}))

x = tf.placeholder("float",None)
x1 = x * 2
y = x1 ** 2

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x: [1,2,3,4,5]}))
