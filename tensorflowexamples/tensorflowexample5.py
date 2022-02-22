import tensorflow as tf

num = tf.Variable(0, name="number")

message = tf.Variable("",name="message")

adder = tf.add(num, tf.constant(1, name="increment"))
oddeven = tf.mod(num, tf.constant(2, name="oddeven"))
assigner = tf.assign(num, adder)

assign_message_odd = tf.assign(message, tf.constant("Is Odd"))
assign_message_even = tf.assign(message, tf.constant("Is Even"))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(100):
        sess.run(assigner)
        if sess.run(oddeven)==0:
            print("%d %s" % (sess.run(num), sess.run(assign_message_even)))
        else:
            print("%d %s" % (sess.run(num), sess.run(assign_message_odd)))


