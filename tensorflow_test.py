# tensorflow needs Python version 3.5.2 (exactly)
# and microsoft visual c++
import tensorflow as tf
import matplotlib.pyplot as plt

# generate first "node" of tensorflow-graph, which outputs a tensor with 100 values from -3 to 3
x = tf.lin_space(-3.,3.,50)

# as you see, this node isn't evaluated yet... you just get the specifications of that node
print(x)

#
g = tf.get_default_graph()

[op.name for op in g.get_operations()]

g.get_tensor_by_name('LinSpace'+':0')

# All operations have to be done within a "session"
sess = tf.Session()

# 2 ways to run a computation
computed = sess.run(x)
hello = tf.constant('hello tensorflow!')
print(sess.run(hello))
print(computed)

computed = x.eval(session=sess)
print(computed)
sess.close()

# 3 ways to explicitly specify the graph which should be computed by session
sess = tf.Session(graph=g)
sess.close()

sess = tf.Session(graph=tf.get_default_graph())
sess.close()

g2 = tf.Graph()
sess = tf.Session(graph=g2)
sess.close()

# way to go: use an interactive session:

sess = tf.InteractiveSession()
# ... now we can evaluate everything "on the fly"
x.eval()
sess.close()

sess = tf.InteractiveSession()

print(x.eval())
# get tensor shape
print(x.get_shape())
print(x.get_shape().as_list())


# example of more complicated graph: generate gauss-curve
mean = 0
sigma = 1.0

z = (tf.exp(tf.negative(tf.pow(x-mean,2.0)/(2.0*tf.pow(sigma,2.0))))*(1.0/(sigma*tf.sqrt(2.0*3.1415))))
plt.figure(1)
plt.plot(x.eval(),z.eval())
plt.draw()

# convolution example
ksize = z.get_shape().as_list()[0]
# build kernel
z_2d = tf.matmul(tf.reshape(z,[ksize,1]),tf.reshape(z,[1,ksize]))

# plot kernel
plt.figure(2)
plt.imshow(z_2d.eval())
plt.draw()

from scipy import ndimage
import numpy as np
f = ndimage.imread("C:\\Users\\NiWa\\PycharmProjects\\pygame_test\\face.png").astype('float32')
img4d = f.reshape([1,f.shape[0],f.shape[1],3])
img4d_tf = tf.reshape(np.mean(f,axis=2),[1,f.shape[0],f.shape[1],1])
print(img4d_tf.get_shape().as_list())
z_4d = tf.reshape(z_2d,[ksize,ksize,1,1])

convolved = tf.nn.conv2d(img4d_tf,z_4d,strides=[1,1,1,1],padding='SAME')
res = convolved.eval().reshape(f.shape[0],f.shape[1])
plt.figure(3)
print(res.shape)
plt.imshow(res,cmap='gray')
plt.show()

sess.close()