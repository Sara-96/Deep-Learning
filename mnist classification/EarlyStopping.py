#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score


# In[4]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
L_train = mnist.train.labels
X_test = mnist.test.images
L_test = mnist.test.labels


# In[ ]:





# In[6]:


tf.reset_default_graph()

X_Ttrain = X_train[0:49499,:]
X_val = X_train[49500:54999,:]
L_Ttrain = L_train[0:49499,]
L_val = L_train[49500:54999,]

lr = 0.5
Iter = 495
BatchSize = 100

ACC_val = np.zeros([Iter])
ACC_train = np.zeros([Iter])
L_val = np.zeros([Iter])
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
filewriter = tf.summary.FileWriter("finall/06")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()    
tf.summary.FileWriter('finall/06',sess.graph)

Loss_val = 10
min_loss = 10
accuracy_val = 0
for j in range(100):  
    
    if j>10:
        if Loss_val>min_loss:
            break
    min_loss = Loss_val       
    
    for i in range(Iter):
        batch_X, batch_Y = mnist.train.next_batch(BatchSize)
        sess.run(optimizer,feed_dict={x: batch_X, y: batch_Y})
       
        b=(sess.run(merge,feed_dict={x: batch_X, y: batch_Y}))

        filewriter.add_summary(b, i)
        accuracy_train,loss_train = sess.run((acuracy_train,cross_entropy),feed_dict={x:batch_X,y:batch_Y})
        ACC_train[i] = accuracy_train
        L_train[i] = loss_train
        
        if i%50==0:  
            print ("step %5i in epoc %3i train_acc: %g - train_cross_entropy: %g" %(i,j,accuracy_train,loss_train))

    [accuracy_val,Loss_val] = sess.run([acuracy_train,cross_entropy], feed_dict={x: X_test, y: L_test})
    print ("epoc %3i validation_acc: %g - validation_loss: %g" %(j,accuracy_val,Loss_val))


# In[7]:


#[accuracy_testوLoss_test] = sess.run([acuracy_train,cross_entropy], feed_dict={x: X_test, y: L_test})
[accuracy_test,Loss_test] = sess.run([acuracy_train,cross_entropy], feed_dict={x: X_test, y: L_test})
print ("The accuracy of test:"+str(accuracy_test))
print ("The Loss of test:"+str(Loss_test))


# In[ ]:




