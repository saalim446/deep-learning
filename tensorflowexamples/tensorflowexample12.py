import tensorflow as tf

a = tf.Variable([1,2,3,4,5],name="Variable_1")
# graph = tf.Graph()

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())

     # print(graph.as_graph_def())

     tf.io.write_graph(sess.graph,"./logs","graph.pbtxt")


