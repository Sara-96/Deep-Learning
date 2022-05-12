#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score


# In[10]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
L_train = mnist.train.labels
X_test = mnist.test.images
L_test = mnist.test.labels


# In[28]:


tf.reset_default_graph()
BatchSize = 55000
lr = 0.5
Iter = 100
ACC_test = np.zeros([Iter])
ACC_train = np.zeros([Iter])
L_test = np.zeros([Iter])
L_train = np.zeros([Iter])
x = tf.placeholder(dtype=tf.float32,shape=(None,784),name="input")
y = tf.placeholder(dtype=tf.float32,shape=(None,10),name="label")

with tf.name_scope(name="Hidden_layer"):
    w1 = tf.Variable(tf.random_normal([784, 10], mean=0, stddev=0.2),name="W")
    b1 = tf.Variable(tf.zeros([10]),name="B")
    
    hidden_layer1_input = tf.matmul(x,w1)+b1
    sigmoid1 = tf.sigmoid(hidden_layer1_input)


with tf.name_scope("output_layer"):
    w2 = tf.Variable(tf.random_normal([10, 10], mean=0, stddev=0.2),name="W")
    b2 = tf.Variable(tf.zeros([10]),name="B")

    outputlayer_input = tf.matmul(sigmoid1,w2)+b2
    output = tf.nn.softmax(outputlayer_input)

with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output))

      
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    acuracy_train = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32),name="accuracy_train")   
    

tf.summary.scalar("Cross_Entropy",cross_entropy)
tf.summary.scalar("aacuracy_train",acuracy_train)
tf.summary.histogram("Wights_Hidden_layer",w1)
tf.summary.histogram("Wights_Output_layer",w2)

merge = tf.summary.merge_all()
filewriter = tf.summary.FileWriter("finall/14")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()    
tf.summary.FileWriter('finall/14',sess.graph)

for i in range(Iter):
    batch_X, batch_Y = mnist.train.next_batch(BatchSize)
    sess.run(optimizer,feed_dict={x: batch_X, y: batch_Y})
       
    b=(sess.run(merge,feed_dict={x: batch_X, y: batch_Y}))

    filewriter.add_summary(b, i)
    batch_Xtest, batch_Ytest = mnist.test.next_batch(BatchSize)
    accuracy_test,loss_test = sess.run([acuracy_train,cross_entropy],feed_dict={x:batch_Xtest,y:batch_Ytest})
    ACC_test[i] = accuracy_test
    L_test[i] = loss_test
    c=(sess.run(merge,feed_dict = {x: batch_Xtest, y: batch_Ytest}))
    filewriter.add_summary(c, i)
    accuracy_train,loss_train = sess.run((acuracy_train,cross_entropy),feed_dict={x:batch_X,y:batch_Y})
    ACC_train[i] = accuracy_train
    L_train[i] = loss_train
    if i%10==0:  
        print ("step %5i train_acc: %g - train_cross_entropy: %g / test_acc: %g - test_cross_entropy: %g"
               %(i,accuracy_train,loss_train,accuracy_test,loss_test))


# In[ ]:





# In[ ]:




