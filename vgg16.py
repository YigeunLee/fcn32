import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
import skimage
import scipy
import os
import glob
import sys
import random

classes = 3 # yes,no

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25
batch_size = 4
pool3_fmaps = conv2_fmaps
pool5_map = 7

he_init = tf.contrib.layers.variance_scaling_initializer()
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, 224, 224,3] ,name="X")
    dense_y = tf.placeholder(tf.int32, shape=[None,3], name="dense_y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

with tf.variable_scope("feature"):
    conv1 = tf.layers.conv2d(X, filters=64, kernel_size=3,
                             strides=1, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3,
                             strides=1, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv2")
    pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1,2,2, 1], padding="SAME")

    
    conv3 = tf.layers.conv2d(pool1, filters=128, kernel_size=3,
                             strides=1, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv3")
    conv4 = tf.layers.conv2d(conv3, filters=128, kernel_size=conv2_ksize,
                             strides=1, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv4")
    pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1,2,2, 1], padding="SAME")

    
    conv5 = tf.layers.conv2d(pool2, filters=256, kernel_size=3,
                             activation=tf.nn.relu, name="conv5")
    conv6 = tf.layers.conv2d(conv5, filters=256, kernel_size=3,
                             activation=tf.nn.relu, name="conv6")
    conv7 = tf.layers.conv2d(conv6, filters=256, kernel_size=3,
                             strides=1, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv7")
    pool3 = tf.nn.max_pool(conv7, ksize=[1,2, 2, 1], strides=[1, 2,2, 1], padding="SAME")


    conv8 = tf.layers.conv2d(pool3, filters=512, kernel_size=3,
                             strides=1, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv8")
    conv9 = tf.layers.conv2d(conv8, filters=512, kernel_size=3,
                             strides=1, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv9")
    conv10 = tf.layers.conv2d(conv9, filters=512, kernel_size=3,
                             strides=1, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv10")
    pool4 = tf.nn.max_pool(conv10, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding="SAME")


    conv11 = tf.layers.conv2d(pool4, filters=512, kernel_size=3,
                             strides=1, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv11")
    conv12 = tf.layers.conv2d(conv11, filters=512, kernel_size=3,
                             strides=1, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv12")
    conv13 = tf.layers.conv2d(conv12, filters=512, kernel_size=3,
                             strides=1, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv13")
    pool5 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1,2,2, 1], padding="SAME")



with tf.name_scope("vgg_dnn"):
    pool5_flat = tf.reshape(pool5, shape=[-1, pool5_map * pool5_map * 512])
    dnn = tf.layers.dense(pool5_flat,2048, activation=tf.nn.relu, name='fc1')
    dnn = tf.layers.dropout(dnn, tf.Variable(0.08), training=training)
    dnn = tf.layers.dense(dnn,2048,activation=tf.nn.relu, name='fc2')
    dnn = tf.layers.dropout(dnn, tf.Variable(0.08), training=training)
    dnn = tf.layers.dense(dnn,classes, name='fc3')
    dense_logit = tf.nn.softmax(dnn)
    dense_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=dnn, labels=dense_y)
    dense_loss = tf.reduce_mean(dense_xentropy)
    dense_optimizer = tf.train.AdamOptimizer(tf.Variable(0.0001))
    dense_training_op = dense_optimizer.minimize(dense_loss)

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def get_validation():
    inputs = []

    for fname in os.listdir(path=r'E:\개인 프로젝트\텐서플로\test_data\test'):
        pathname = os.path.join(r'E:\개인 프로젝트\텐서플로\test_data\test',fname)
        
        if os.path.isdir(pathname) == False:
            if pathname.find('_label') == -1:
                input_image = scipy.misc.imresize(scipy.misc.imread(pathname), (224,224))
                inputs.append(input_image)
                break
        
        
    return np.array(inputs,dtype=np.float32) / 255.0

def dense_get_batches():
    inputs = []
    labels = []
    batch_cnt = 0
    bg_color = np.array([0, 0, 0])# background color is black

    files = os.listdir(path=r'E:\개인 프로젝트\텐서플로\test_data')
    
    #for fname in os.listdir(path=r'E:\개인 프로젝트\텐서플로\test_data'):
    
    while batch_cnt < batch_size:
        pathname = os.path.join(r'E:\개인 프로젝트\텐서플로\test_data',random.choice(files))
        if os.path.isdir(pathname) == False:
            if pathname.find('_label') == -1:
                input_image = scipy.misc.imresize(scipy.misc.imread(pathname), (224,224))
                inputs.append(input_image)

                if pathname.find('cat') != -1:
                    labels.append([0,0,1])
                else:
                    labels.append([0,1,0])
                
                batch_cnt += 1

    return np.array(inputs,dtype=np.float32) / 255.0 , labels

best_loss_val = np.infty
check_interval = 100
checks_since_last_progress = 0
n_epochs = 1000

with tf.Session() as sess:
    init.run()
    #saver.restore(sess,r'E:\개인 프로젝트\텐서플로\vgg16_fcn.ckpt')
    
    for epoch in range(n_epochs):      
        X_batch, dense_y_batch = dense_get_batches()
        sess.run(dense_training_op, feed_dict={X: X_batch, dense_y: dense_y_batch,training: True})

        if epoch % check_interval == 0:
            loss_val = dense_loss.eval(feed_dict={X: X_batch, dense_y: dense_y_batch,training: False})
            print('loss:' + str(loss_val) + ' ' + 'epoch:' + str(epoch))
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                checks_since_last_progress = 0
                best_model_params = get_model_params()
            else:
                checks_since_last_progress += 1
            saver.save(sess,r'E:\개인 프로젝트\텐서플로\vgg16_fcn.ckpt')
         
    x_val = get_validation()
    #np.set_printoptions(threshold=sys.maxsize)
    output = sess.run(dense_logit, feed_dict={X: x_val,training: False})
    print(output)
