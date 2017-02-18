import theano
from theano import tensor
import tensorflow as tf
import keras

a = tensor.dscalar()
b = tensor.dscalar()
c = a + b
f = theano.function([a,b], c)
result = f(1.5, 2.5)
print(result)


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add = tf.add(a, b)

sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)

print(c)