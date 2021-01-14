""" Multilayer Perceptron.
A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#mnist = input_data.read_data_sets('P:/mlp_test/Playground/mnist_c_test/data/', one_hot=True)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import json

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 100
display_step = 1



# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float32", [None, n_input])
Y = tf.placeholder("float32", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1']) # tf.add
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']) #, biases['b2']
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) #GradientDescentOptimizer
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                              Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

    #prints trainable vars
    #is identical to my extracted weights
    #test_weights=[]
    #tvars = tf.trainable_variables()
    #print(tvars)
    #tvars_vals = sess.run(tvars)
    #for var, val in zip(tvars, tvars_vals):
    #   #print("github: ", var.name, val)
    #    test_weights.append({var : val})
    #for elements in tvars_vals:
    #    for e in elements:
    #        test_weights.append(e)
    #        print(e)

    ##is identical to my extracted weights
    #print("W:", weights['h1'].eval())

    h1_weights=[]
    #for h1 in sess.run(weights['h1']): #replace .eval()
    for h1 in weights['h1'].eval():
        for i in h1:
            h1_weights.append(float(i))
    with open('C:/Users/Ben/Desktop/ann_data/datas/h1.json', 'w') as outfile:
        json.dump(h1_weights, outfile)

    h2_weights=[]
    #for h2 in sess.run(weights['h2']):
    for h2 in weights['h2'].eval():
        for j in h2:
            h2_weights.append(float(j))
    with open('C:/Users/Ben/Desktop/ann_data/datas/h2.json', 'w') as outfile:
        json.dump(h2_weights, outfile)

    out_weights=[]
    #for out in sess.run(weights['out']):
    for out in weights['out'].eval():
        for k in out:
            out_weights.append(float(k))
    with open('C:/Users/Ben/Desktop/ann_data/datas/out.json', 'w') as outfile:
        json.dump(out_weights, outfile)

    b1_weights=[]
    #for b1 in sess.run(biases['b1']):
    for b1 in biases['b1'].eval():
        b1_weights.append(float(b1))
    with open('C:/Users/Ben/Desktop/ann_data/datas/b1.json', 'w') as outfile:
        json.dump(b1_weights, outfile)

    b2_weights=[]
    #for b2 in sess.run(biases['b2']):
    for b2 in biases['b2'].eval():
        b2_weights.append(float(b2))
    with open('C:/Users/Ben/Desktop/ann_data/datas/b2.json', 'w') as outfile:
        json.dump(b2_weights, outfile)

    bout_weights=[]
    #for bout in sess.run(biases['out']):
    for bout in biases['out'].eval():
        bout_weights.append(float(bout))
    with open('C:/Users/Ben/Desktop/ann_data/datas/bout.json', 'w') as outfile:
        json.dump(bout_weights, outfile)


    with open('C:/Users/Ben/Desktop/ann_data/mnist_dict.json', 'w') as f:
        json.dump({
            'bias1': b1_weights,
            'bias2': b2_weights,
            'biasout': bout_weights,
            'h1_weights': h1_weights,
            'h2_weights': h2_weights,
            'out_weights': out_weights,
            },
            f
        )
    weightlist = h1_weights + b1_weights + h2_weights + b2_weights + out_weights + bout_weights
    #weightlist = h1_weights + h2_weights + out_weights

    weights = []
    for weight in weightlist:
        weights.append(float(weight))


    count = 0
    for i in weightlist:
        count = count + 1

    #export weights to JSON
    with open('C:/Users/Ben/Desktop/ann_data/mnist.json', 'w') as outfile:
        json.dump(weights, outfile)
    print("number of weights total: ", count)

    print("Optimization Finished!")

    #print(mnist) #some infos

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

