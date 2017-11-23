import tensorflow as tf
import numpy as np

def body(x):
    with tf.name_scope("body"):
        a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100,name="a")
        b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32,name="b")
        c = a + b
        return tf.nn.relu(x + c)

def condition(x):
    with tf.name_scope("condition"):
        return tf.reduce_sum(x) < 100

x = tf.Variable(tf.constant(0, shape=[2, 2]),name="x")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = tf.while_loop(condition, body, [x])
    print(sess.run(result))

    y = tf.Variable(tf.constant(0,shape=[2,2]),name="y")
    y.assign(result)

    # TensorBoard
    hyperparameter = 'run1'
    summary_writer = tf.summary.FileWriter(
        'C:/Users/NiWa/PycharmProjects/pygame_test/tensorboard_logs/test4/' + hyperparameter, tf.get_default_graph())
