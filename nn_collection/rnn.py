# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:07:01 2018

@author: sara.zeng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def weight_variable(shape):
    shape = tf.TensorShape(shape)
    initial_values = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_values)

def bias_variable(shape):
    initial_values = tf.zeros(tf.TensorShape(shape))
    return tf.Variable(initial_values)

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

total = a+b
z = x+y

vec = tf.random_uniform(shape=(3,)) 
# produce a tf.Tensor that generates a random 3-element vector 
# (with values in [0, 1))

sess = tf.Session()

out1 = vec+1
print(sess.run(out1))

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

print(sess.run(total))
print(sess.run({'ab': (a,b), 'total': total}))

print(sess.run(z, feed_dict={x:[1,3], y:2}))
print(sess.run(z, feed_dict={x:[1,3], y:[2]}))
print(sess.run(z, feed_dict={x:[1,3], y:[2,4]}))

# tf.data are prefered for streaming data
#1. convert a dataset to a tf.data.Iterator
#2. call tf.data.Iterator.get_next
#get a runnable tf.Tensor

r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()
sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break

# initializers
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)