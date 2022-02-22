import tensorflow as tf

a = tf.placeholder(tf.int32,shape=(1, 3))
b = tf.placeholder(tf.int32,shape=(1, 3))

c = tf.add(a, b)
c1 = tf.multiply(a, b)
c2 = tf.subtract(a, b)
c3 = tf.divide(a, b)

with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [[1,2,3]],b: [[3,4,5]]}))
    print(sess.run(c1, feed_dict={a: [[1, 2, 3]], b: [[3, 4, 5]]}))
    print(sess.run(c2, feed_dict={a: [[1, 2, 3]], b: [[3, 4, 5]]}))
    print(sess.run(c3, feed_dict={a: [[1, 2, 3]], b: [[3, 4, 5]]}))