import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# create data

X = np.linspace(start=0,stop=10,num=800)
Y = 3*np.sin(X+0.5)+np.random.standard_normal(800)

# define model

inputX = tf.placeholder(tf.float32,name="X")
outputY = tf.placeholder(tf.float32,name="Y")

with tf.name_scope(name="sinus_model"):
    factor = tf.Variable(tf.constant(1,dtype=tf.float32),name="factor")
    phase = tf.Variable(tf.constant(0,dtype=tf.float32),name="phase")
    predictedY = factor*tf.sin(inputX+phase)

# define root mean square error

with tf.name_scope(name="rms_error"):
    cost = tf.sqrt(tf.reduce_mean(tf.square(predictedY-outputY)),name="cost")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    hyperparameter = 'run2'
    summary_writer = tf.summary.FileWriter('C:/Users/NiWa/PycharmProjects/pygame_test/tensorboard_logs/test3/'+hyperparameter,tf.get_default_graph())

    tf.summary.scalar(name="cost",tensor=cost)
    tf.summary.scalar(name="factor",tensor=factor)
    tf.summary.scalar(name="phase",tensor=phase)
    summary_all = tf.summary.merge_all()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss=cost)
    for i in range(200):
        _,s = sess.run([optimizer,summary_all],feed_dict={inputX:X[[i,200+i,400+i,600+i]],outputY:Y[[i,200+i,400+i,600+i]]})
        summary_writer.add_summary(s,i)

    plt.plot(X,Y,'r+',X,factor.eval(sess)*np.sin(X+phase.eval(sess)))
    plt.show()