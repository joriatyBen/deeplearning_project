""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
import json

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf

# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)


names = model.get_variable_names()
print("name:", names)
print('weights:')

weightlist=[]
#single lists for sorting elements to C-algorithmn
hidden0_bias =[]
hidden0=[]
hidden1_bias =[]
hidden1=[]
logits_bias = []
logits=[]


for i in names:
    print(model.get_variable_value(i))
    #weightlist.append(classifier.get_variable_value(i))
    if(i=='dense/bias'):
        hidden0_bias.append(model.get_variable_value(i))
    elif(i=='dense/kernel'):
        for j in model.get_variable_value(i):
            hidden0.append(j)
    elif(i== 'dense_1/bias'):
        hidden1_bias.append(model.get_variable_value(i))
    elif(i== 'dense_1/kernel'):
        for j in model.get_variable_value(i):
            hidden1.append(j)
    elif(i== 'dense_2/bias'):
        logits_bias.append(model.get_variable_value(i))
    elif(i== 'dense_2/kernel'):
        for j in model.get_variable_value(i):
            logits.append(j)

#matrix transponse:
#dense_kernel_transp = [[hidden0[j][i] for j in range(len(hidden0))] for i in range(len(hidden0[0]))]
#dense1_kernel_transp = [[hidden1[j][i] for j in range(len(hidden1))] for i in range(len(hidden1[0]))]
#logits_transp = [[logits[j][i] for j in range(len(logits))] for i in range(len(logits[0]))]

#cumulate lists
weightlist = hidden0 + hidden0_bias + hidden1 + hidden1_bias + logits + logits_bias
#weightlist = dense_kernel_transp + hidden0_bias + dense1_kernel_transp + hidden1_bias + logits_transp + logits_bias
weights = []
for weight in weightlist:
    for w in weight:
        #print(w)
        weights.append(float(w))
print(weights)

count = 0;
for i in weights:
    count = count+1


#export weights to JSON
with open('C:/Users/bajorat_benjamin/Documents/codenv/mnist_examp/esti_mnist.json', 'w') as outfile:
    json.dump(weights, outfile)

print("Testing Accuracy:", e['accuracy'])
print("number of weights total: ", count)
