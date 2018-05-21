
# coding: utf-8

# # Convolutional Neural Network Example
# 
# Build a convolutional neural network with TensorFlow.
# 
# This example is using TensorFlow layers API, see 'convolutional_network_raw' example
# for a raw TensorFlow implementation with variables.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# ## CNN Overview
# 
# ![CNN](http://personal.ie.cuhk.edu.hk/~ccloy/project_target_code/images/fig3.png)
# 
# In[1]:


from __future__ import division, print_function, absolute_import


import tensorflow as tf
from tensorflow.python.training import checkpoint_state_pb2
import matplotlib.pyplot as plt
import numpy as np

import scipy.io as sio
import os
import uuid
import datetime

from sklearn.model_selection import train_test_split




def over_run(num_steps,dir_name,mat_name,tensor_out_dir):



    print(mat_name)
    file = os.path.join(dir_name,mat_name)
    print(file)

    model_dir_name = 'chirp_model'





    print(num_steps)


    # mat_contents = sio.loadmat('chirp_4D.mat')
    mat_contents = sio.loadmat(file)
    chirp_4d = mat_contents['full_array_nc'].astype('float32')
    max_x = mat_contents['min_x_dim']
    max_y = mat_contents['dim_y']
    y = mat_contents['y']
    test_description = mat_contents['test_description']
    test_description = str(test_description[0])
    print (type(test_description))



    print('Working on:' + test_description + '\n')


    #print (mat_contents)
    #print (chirp_4d)
    print (max_x)
    print (max_y)
    #print (chirp_4d.shape)

    #print (np.unique(labels).size)

    num_classes = np.unique(y).size

    #Here I am finding the number of chirplets in our "deck"
    #The top of the convnet will be adjusted as such.



    #chirp_4d = chirp_4d[:,:,0,:]
    #print(chirp_4d.shape)
    chirp_4d_orig = chirp_4d

    num_channels = chirp_4d[0,0,:,0].size
    if num_channels == 0:
        num_channels = 1
    print(num_channels)

    print (chirp_4d[10,:,:,0])




    #Show some of the chirplet images. 


    # for k in range(0,num_channels):
        #print(k)
        #plt.imshow(chirp_4d_orig[:,:,k,10],cmap='gray')
        #plt.show()
        #The above matches the following in Matlab. 
        #imagesc(full_array_nc(:,:,1,11))


    #plt.imshow(np.reshape(chirp_4d[:,:,0,0], [max_x, max_y]), cmap='gray')



    print (y.shape)
    print(y.dtype)
    print(chirp_4d.dtype)
    print (np.reshape(y,[-1,1]).shape)
    print (np.squeeze(np.reshape(y,[-1,1])).shape)

    labels = np.squeeze(np.reshape(y,[-1,1]))



    print(chirp_4d.shape)
    chirp_4d = np.moveaxis(chirp_4d,3,0)
    print(chirp_4d.shape)

    print(labels.shape)
    print(labels[0:10])







    # Training Parameters
    learning_rate = 0.001
    #num_steps = 1000
    batch_size = 10

    np.random.seed(1)
    tf.set_random_seed(1)



    # Network Parameters
    num_input = 784 # MNIST data input (img shape: 28*28)
    #num_classes = 10 # MNIST total classes (0-9 digits)
    dropout = 0.25 # Dropout, probability to drop a unit


    # In[82]:


    # Create the neural network
    def conv_net(x_dict, n_classes, dropout, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs
            x = x_dict['images']

            # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(x, shape=[-1, max_x, max_y, num_channels])
            
            
            
            
            #Create the 4D chirplet input vector. 
            #x = tf.reshape(x, shape=[-1, 28, 28, 1])
            
            print (x.shape)
            
            
            

            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv2)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)
            
            


        return out




    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode):
        
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
        logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)
        
        # Predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)
        
        
        
        
        
        
        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
            
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
        
        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)


        constant = 8
        tf.summary.scalar('Accuracy_Train', acc_op[1])

        tf.summary.scalar('Loss_Train', loss_op)
        tf.summary.scalar('Test', constant)




        eval_summary_hook = tf.train.SummarySaverHook(
                                save_steps=1,
                                #output_dir= self.job_dir,
                                summary_op=tf.summary.merge_all())
        
        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=pred_classes,
          loss=loss_op,
          train_op=train_op,
          eval_metric_ops={'Accuracy_Test': acc_op},
          evaluation_hooks=[eval_summary_hook])

        return estim_specs





    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs = 0  # Save checkpoints every 20 minutes.
    )

    #folder_name = datetime.datetime.now().strftime("%d_%H") + test_description
    folder_name =  test_description

    directory = os.path.join(tensor_out_dir,model_dir_name,str(num_steps),folder_name,datetime.datetime.now().strftime("%Y_%m_%d %H_%M_%S"))



    # Build the Estimator
    model = tf.estimator.Estimator(model_fn,model_dir=directory)




    # Define the input function for training
    #input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={'images': mnist.train.images}, y=mnist.train.labels,
    #    batch_size=batch_size, num_epochs=None, shuffle=True)


    print(chirp_4d.shape)
    print(labels.shape)


    #plt.imshow(chirp_4d[10,:,:,0],cmap='gray')

    #The above matches the following in Matlab. 
    #imagesc(full_array_nc(:,:,1,11))


    #plt.imshow(np.reshape(chirp_4d[:,:,0,0], [max_x, max_y]), cmap='gray')
    #plt.show()




    #Use the scikit learn train_test split function to split our data for training and testing. 
    X_train,X_test,y_train,y_test = train_test_split(chirp_4d,labels,test_size=0.2)



    # X_train1,X_test1,y_train1,y_test1 = train_test_split(chirp_4d,labels,test_size=0.2,random_state=1)


    # print(np.array_equal(X_train,X_train1))
    # print(np.array_equal(X_test,X_test1))
    # print(np.array_equal(y_train,y_train1))
    # print(np.array_equal(y_test,y_test1))

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)






    # Define the input function for training with the 4D chirplets.
    input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x={'images': X_train}, y=y_train,
        batch_size=batch_size, num_epochs=None, shuffle=False)




    tf.set_random_seed(2)


    #tensors_to_log = {"acc": "out"}
    #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)




    # Train the Model
    # model.train(input_fn, steps=num_steps)




    # Evaluate the Model
    # Define the input function for evaluating
    #input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={'images': mnist.test.images}, y=mnist.test.labels,
    #    batch_size=batch_size, shuffle=False)


    # Evaluate the 4D Chirplet Model
    # Define the input function for evaluating

    input_fn_test = tf.estimator.inputs.numpy_input_fn(
        x={'images': X_test}, y=y_test,
        batch_size=batch_size, shuffle=False)


    # Use the Estimator 'evaluate' method



    experiment = tf.contrib.learn.Experiment(
    estimator=model,
    train_input_fn=input_fn_train,
    eval_input_fn=input_fn_test,
    train_steps=num_steps,
    eval_steps=None, # evaluate runs until input is exhausted
    eval_delay_secs=0, 
    train_steps_per_iteration=50 #Essentially runs testing every N steps of the train_steps
    )

    experiment.continuous_train_and_eval()  






    return directory


   