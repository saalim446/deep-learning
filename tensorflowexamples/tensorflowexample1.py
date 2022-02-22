

import tensorflow as tf

hello = tf.constant("Hello World!")
hello1 = tf.constant("Hello World!")
sess = tf.Session()
# session is for executing a tensor or a graph -> flow,subgraph
# session can be for executing a flow/ subgraph
# subgraphs are subset or a graph
# a flow can be one tensor or a graph ,graph which can be a flow or a subgraph
print(sess.run(hello))

