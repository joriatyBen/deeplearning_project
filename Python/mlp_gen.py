from random import seed
from random import random
from math import exp
import math
import itertools
import csv
import os
import numpy as np
from csv import reader
import operator
import json
import six.moves.urllib.request as request


PATH = "C:/Users/bajorat_benjamin/Documents/codenv"
# Fetch and store Training and Test dataset files
PATH_DATASET = PATH + os.sep + "dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "iris_training.csv"
FILE_TEST = PATH_DATASET + os.sep + "iris_test.csv"
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"

def download_dataset(url, file):
    if not os.path.exists(PATH_DATASET):
        os.makedirs(PATH_DATASET)
    if not os.path.exists(file):
        data = request.urlopen(url).read()
        with open(file, "wb") as f:
            f.write(data)
            f.close()

#Initialize generic MLP architecture
def init_network(n_input, n_output, bias, *hidden_nodes):
    network = list()
    n_hidden = (hidden_nodes)
    #global b
    #b = bias
    #check for hidden layer
    try:
        #hidden_layer = [weights, nodes]
        hidden_layer = [{'weights':[random() for i in range(n_input + bias)]} for i in range(n_hidden[0])]
        network.append(hidden_layer)
    except:
        output_layer = [{'weights':[random() for i in range(n_input + bias)]} for i in range(n_output)]
        network.append(output_layer)

    j=0
    for nodes in n_hidden:
        try:
            hidden_layer = [{'weights':[random() for i in range(n_hidden[j] + bias)]} for i in range(n_hidden[j+1])]
            network.append(hidden_layer)
            j+=1
        except:
            output_layer = [{'weights':[random() for i in range(n_hidden[-1] + bias)]} for i in range(n_output)]
            network.append(output_layer)

    return network

#weight producer
def activate(weights, neuron_inputs, bias):
    if(bias==1): # bias existing
        activation = weights[-1] #bias weight,
        for i in range(len(weights)-1): #cuts out the bias weight lane
            activation += weights[i] * neuron_inputs[i]
    else: #bias not existing
        activation = 0
        for i in range(len(weights)):
            activation += weights[i] * neuron_inputs[i]
    return activation

#activation function types
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

def tanh(activation):
    return (2.0 / (1.0 + exp(-2*activation)))-1

def relu(activation):
    if(activation <= 0):
        return 0
    else:
        return activation

def leaky_relu(activation):
    if(activation <= 0):
        return 0.01*activation
    else:
        return activation

def sign(activation):
    if(activation <= 0):
        return 0
    else:
        return 1

#forward propagation
def forward_propagate(network, bias ,input_data, a_function, *args):
    inputs = input_data
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs, bias)
            #creates a new dictionary variable with the result of the a_function
            args = activation
            neuron['output'] = a_function(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs #this return is basically the output of the last neurons

#activation function derivatives
def sigmoid_derivative(output):
    return output * (1.0 - output)

def tanh_derivative(output):
    return 1 - ((2.0 / (1.0 + exp(-2*output)))-1)**2

def relu_derivative(output):
    if(output > 0):
        return 1
    else:
        return 0

def sign_derivative(output):
    if(output == 0):
        return None
    else:
        return 0

#backprop
def backward_propagate_error(network, expected, func_derivative, *args):
    for i in reversed(range(len(network))):
        layer = network[i] #list with reversed layers from output to input (=backwards)
        errors = list()
        if i != len(network)-1: #last layer
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            args = neuron['output']
            neuron['delta'] = errors[j] * func_derivative(args)
            #neuron['output'] is defined in forward_propagate

def update_weights(network, bias, train_dataset_row, l_rate):
    trained_network=[]
    for i in range(len(network)):
        inputs = train_dataset_row[:-1] #excluding the output, adjust when there is more than 1 Output neuron!!
        if(bias == 1): #bias existing
            if(i != 0):
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta'] #Biasneuron
        else:
            if(i != 0): #bias not existing
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
    trained_network = network
    return trained_network


# Train network for fixed number of epochs
def train_network(network, bias,train_dataset, l_rate, n_epoch, n_outputs, a_function, func_derivative, *args):
    final_network=[]
    expected =[]
    e = 0.001 #error-rate
    sum_prerror =0
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train_dataset: # last column in row is output value
            outputs = forward_propagate(network, bias ,row, a_function)

            if(row[-1]==0):
                expected=[1,0,0]
            if(row[-1]==1):
                expected=[0,1,0]
            if(row[-1]==2):
                expected=[0,0,1]

            #expected = [0 for i in range(n_outputs)]
            #expected[int(row[-1])] = 10 # tried 10 for a higher rate, its working way better than 1

            sum_error += sum([abs(0.5*(outputs[i]-expected[i])**2) for i in range(len(outputs))])
            backward_propagate_error(network, expected, func_derivative)
            update_weights(network, bias,row , l_rate)
            final_network = update_weights(network, bias,row , l_rate)
        difference = abs(sum_prerror - sum_error)
        print("error of previous epoche: ", sum_prerror)
        print("error of current epoche: ", sum_error)
        sum_prerror = sum_error
        if(difference < e):
            print("MLP is trained in {} epochs".format(epoch))
            break
    return final_network

# predict after training
def predict(network,bias ,dataset_row, a_function):
    outputs = forward_propagate(network, bias, dataset_row, a_function)
    return outputs #.index(max(outputs))


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(bias, train_dataset, test_dataset, l_rate, n_epoch, a_function, func_derivative, *n_hidden_nodes):
    final_network=[]
    n_inputs = len(train_dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in train_dataset]))
    network = init_network(n_inputs, n_outputs, bias, *n_hidden_nodes)
    #train_network(network, bias, train_dataset, l_rate, n_epoch, n_outputs, a_function, func_derivative)
    final_network= train_network(network, bias,train_dataset, l_rate, n_epoch, n_outputs, a_function, func_derivative)
    #print('trained network: ',final_network )
    network = final_network
    for row in test_dataset:
        prediction = predict(network, bias, row, a_function)
        index, value = max(enumerate(prediction), key=operator.itemgetter(1))
        if(index == 0):
            print('Decision value: {} Iris Setosa --> 0: '.format(value))
        if(index == 1):
            print('Decision value: {} Iris Versicolor --> 1: '.format(value))
        if(index == 2):
            print('Decision value: {} Iris Virginica --> 2: '.format(value))
    return final_network #prediction

def main():
    #global bias var
    bias=0
    l_rate = 0.0001
    n_epoch = 1000

    #download_dataset(URL_TRAIN, FILE_TRAIN)
    #download_dataset(URL_TEST, FILE_TEST)

    # get train dataset
    train_dataset = []
    with open('iris_training.csv', 'r') as f:
        csv_reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        next(f, None)
        for row in csv_reader:
            #print(row[0:-1])
            train_dataset.append(row)
        #print(train_dataset)

    # get test dataset
    test_dataset = []
    with open('iris_test.csv', 'r') as f:
        csv_reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        next(f, None)
        for row in csv_reader:
            #print(row[0:-1])
            test_dataset.append(row)
        #print(test_dataset)

    n_inputs = len(train_dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in train_dataset]))
    print('Inputs: ',n_inputs)
    print('Outputs:',n_outputs)

    #create network
    #network = init_network(n_inputs, n_outputs, 0, 2, 2)
    #print(network)

    final_network=[]
    #back_propagation(train_dataset, test_dataset, l_rate, n_epoch, relu, relu_derivative)
    final_network=back_propagation(bias,train_dataset, test_dataset, l_rate, n_epoch, relu, relu_derivative,4,4)
    prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                    [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                    [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Sentosa

    print('making a prediction: ')
    for dataset_row in prediction_input:
        prediction=predict(final_network, bias, dataset_row, relu)
        index, value = max(enumerate(prediction), key=operator.itemgetter(1))
        if(index == 0):
            print('Decision value: {} Iris Setosa --> 0: '.format(value))
        if(index == 1):
            print('Decision value: {} Iris Versicolor --> 1: '.format(value))
        if(index == 2):
            print('Decision value: {} Iris Virginica --> 2: '.format(value))

    elements=[]
    for n in final_network:
        for element in n:
            elements.append(element['weights'])
    print(elements)

    weightlist=[]
    for e in elements:
        for weights in e:
            weightlist.append(weights)
    print(weightlist)
    count = 0
    for i in weightlist:
        count = count + 1
    print(count)

    with open('C:/Users/bajorat_benjamin/Documents/codenv/mlp_gen/datas.json', 'w') as outfile:
        json.dump(weightlist,outfile)


if __name__ == "__main__":
    main()
