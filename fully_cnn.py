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

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def get_deconv_initiailizer(in_channels,out_channels,kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size,:kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((kernel_size,kernel_size,in_channels, out_channels),
                      dtype='float32')

    #weight[range(in_channels),range(out_channels),:,:] = filt

    for i in range(in_channels):
        weight[:, :, i, i] = filt
    
    return tf.constant_initializer(weight)

def get_zero_init(in_channels,out_channels,kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    weight = np.zeros((in_channels,out_channels,kernel_size, kernel_size),
                      dtype='float32')

    return tf.constant_initializer(weight)

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

pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5

n_outputs = 10

kernel_size = 64
strides = 32

output_size = strides

reset_graph()

he_init = tf.contrib.layers.variance_scaling_initializer()
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, 224, 224,3] ,name="X")
    y = tf.placeholder(tf.float32, shape=[None,output_size,output_size,classes], name="Y") # batch_size,height,width

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
    
pool5_map = 7

with tf.name_scope("fc"):
    fc1 = tf.layers.conv2d(pool5,filters=2048, kernel_size=pool5_map,strides=1,
                           kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                           kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4),
                           activation=tf.nn.relu,
                           padding="VALID", name="conv_fc1")
    fc1 = tf.layers.dropout(fc1, tf.Variable(0.05), training=training)

    fc2 = tf.layers.conv2d(fc1, filters=2048, kernel_size=1,
                           kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4),
                           strides=1,
                           padding="VALID",
                           activation=tf.nn.relu,
                         name="conv_fc2")
    fc2 = tf.layers.dropout(fc2, tf.Variable(0.05), training=training)

    fc3 = tf.layers.conv2d(fc2, filters=classes,
                         kernel_initializer = tf.keras.initializers.Zeros(),
                         kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4),
                         kernel_size=1,
                         strides=1,
                         name="conv_fc3")
    pred_softmax = tf.nn.softmax(fc3)

with tf.name_scope("transpose_output"):
    trans = tf.layers.conv2d_transpose(fc3,classes,
                                       kernel_initializer = get_deconv_initiailizer(classes,classes,kernel_size),
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       name="transpose_fc",
                                       padding='SAME')
    
    test_softmax = tf.nn.softmax(trans)
with tf.name_scope("train"):
    reshaped_logits = tf.reshape(trans, shape=(-1,classes))
    reshaped_labels = tf.reshape(y,shape=(-1,classes))
    
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=reshaped_logits, labels=reshaped_labels)
    loss = xentropy#tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(tf.Variable(0.0001))
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="conv_fc[123]|transpose_fc") 
    training_op = optimizer.minimize(loss,var_list=train_vars)

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

best_loss_val = np.infty
check_interval = 100
checks_since_last_progress = 0
n_epochs = 3000

dog = [65,0,128]
cat = [64,129,65]

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}
def get_validation():
    inputs = []
    labels = []
    
    bg_color = np.array([0, 0, 0])# background color is black
    for fname in os.listdir(path=r'E:\개인 프로젝트\텐서플로\test_data\test'):
        pathname = os.path.join(r'E:\개인 프로젝트\텐서플로\test_data\test',fname)
        
        if os.path.isdir(pathname) == False:
            if pathname.find('_label') == -1:
                input_image = scipy.misc.imresize(scipy.misc.imread(pathname), (224,224))
                inputs.append(input_image)
                break
        
        
    return np.array(inputs,dtype=np.float32) / 255.0


def get_result(width,height,softmax):
    inputs = []
    labels = []
    
    bg_color = np.array([0, 0, 0])# background color is black
    seg = np.zeros((width,height,3),dtype=np.int)
    #print(softmax.shape)
    for y in range(0,softmax.shape[1]):
        for x in range(0,softmax.shape[2]):
            idx = np.argmax(softmax[0,x,y])
            #print(idx)
            if idx == 2:
                seg[x,y] = dog
            elif idx == 1:
                seg[x,y] = cat
  
    return seg
batch_size = 4
def get_batches():
    inputs = []
    labels = []
    batch_cnt = 0
    bg_color = np.array([0, 0, 0])# background color is black

    files = os.listdir(path=r'E:\개인 프로젝트\텐서플로\test_data')
        
    while batch_cnt < batch_size:
        pathname = os.path.join(r'E:\개인 프로젝트\텐서플로\test_data',random.choice(files))
        #print(pathname)
        if os.path.isdir(pathname) == False:
            if pathname.find('_label') == -1:
                input_image = scipy.misc.imresize(scipy.misc.imread(pathname), (224,224))
                inputs.append(input_image)

                bg_path = os.path.splitext(pathname)[0] + '_label.jpg'
                label_image = scipy.misc.imresize(scipy.misc.imread(bg_path),  (output_size,output_size))
                seg = np.zeros((label_image.shape[0],label_image.shape[1],classes),dtype=np.int)
                seg[(label_image == bg_color).all(axis=2)] = [1,0,0]
                seg[(label_image == dog).all(axis=2)] = [0,1,0]
                seg[(label_image == cat).all(axis=2)] = [0,0,1]
                labels.append(seg)
                #print(pathname)
                batch_cnt += 1
            

    return np.array(inputs,dtype=np.float32) / 255.0 , np.array(labels)

with tf.Session() as sess:
    init.run()
    gr = tf.get_default_graph()
    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES , scope="feature")
    #print(reuse_vars)
    #original_saver = tf.train.Saver(reuse_vars) # saver to restore the original model
    #original_saver.restore(sess,r'E:\개인 프로젝트\텐서플로\vgg16_fcn.ckpt')
    saver.restore(sess,r'E:\개인 프로젝트\텐서플로\vgg16_fcn_2.ckpt')
    '''
    for epoch in range(n_epochs):      
        X_batch, y_batch = get_batches()
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch,training: True})

        if epoch % check_interval == 0:
            print('softmax:' + str(sess.run(pred_softmax,feed_dict={X: X_batch,training: False})))
            #loss_val = loss.eval(feed_dict={X: X_batch, y: y_batch})
            #print('loss:' + str(loss_val) + ' ' + 'epoch:' + str(epoch))
            #if loss_val < best_loss_val:
            #    best_loss_val = loss_val
            #    checks_since_last_progress = 0
            #    best_model_params = get_model_params()
            # 
            #else:
            #    checks_since_last_progress += 1
            saver.save(sess,r'E:\개인 프로젝트\텐서플로\vgg16_fcn_2.ckpt')
    '''
    x_val = get_validation()
    #np.set_printoptions(threshold=sys.maxsize)
    output = sess.run(test_softmax, feed_dict={X: x_val,training: False})
    print('softmax:' + str(sess.run(pred_softmax,feed_dict={X: x_val,training: False})))
    #result_seg_image = get_result(224,224,output)
    plt.imshow(x_val[0,:,:,:], cmap=plt.cm.gray)
    plt.show()
    seg_img = get_result(32,32,output)
    plt.imshow(seg_img[:,:,:])
    plt.show()
    '''
    plt.imshow(output[0,:,:,:])
    plt.show()
    plt.imshow(output[0,:,:,2])
    plt.show()
    '''




