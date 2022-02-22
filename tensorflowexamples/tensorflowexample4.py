import tensorflow
import tensorflow as tf

state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

with tf.Session()as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(state))
    for i in range(3):
        sess.run(update)
        print(sess.run(state))