from math import exp
import json
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = input_data.read_data_sets("P:/mlp_test/Playground/mnist_c_test/data", one_hot=True)


# Network Parameters
h1 = 256 # 1st layer number of neurons
h2 = 256 # 2nd layer number of neurons
input = 784 # MNIST data input (img shape: 28*28)
out = 10 # MNIST total classes (0-9 digits)

#TESTIMAGES
def get_mnist_image():
    test_image=[]
    for img in mnist.test.images:
        #img = (np.reshape(img, (28, 28)) * 255).astype(np.float32)
        img = (np.reshape(img, (784)) * 255).astype(np.float32)
        image = np.where(img > 0, 1, 0)
        for i in image:
            test_image.append(i)
    print(test_image)
    return test_image


#TESTLABELS
def get_mnist_label():
    test_label=[]
    label = None
    for label in mnist.test.labels:
        counter = 0
        for i in label:
            if(i == 1):
                label = counter
                #print("label: ", label)
                test_label.append(label)
            counter = counter + 1
    return test_label


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


def main():

    batch_xs, batch_ys = mnist.test.next_batch(2)
    #two_as_array = gen_image(batch_xs[0])
    two_vec = batch_xs[0]
    x = batch_ys[0]
    print(x)
    reshape_vec = (np.reshape(two_vec, (28, 28)) * 255).astype(np.float32)
    reshape_vec = np.where(reshape_vec >0, 1, 0)
    print(reshape_vec)
    plt.show(reshape_vec)

    with open('C:/Users/bajorat_benjamin/Documents/codenv/mnist_examp/mnist_dict.json', 'r') as infile:
        data = json.load(infile)

    h1_res = 256 * [0]
    for pix in two_vec:
        for weight_idx in range(h1):
            curr_weight = data['h1_weights'][weight_idx]
            h1_res[weight_idx] += curr_weight * pix

    # add bias
    for idx in range(len(h1_res)):
        h1_res[idx] += data['bias1'][idx]

    # and apply relu
    for idx in range(len(h1_res)):
        h1_res[idx] = relu(h1_res[idx])
    print(h1_res)
    print(range(len(h1_res)))

    h2_res = 256 *[0]
    for res in h1_res:
        for weight_idx in range(h2):
            curr_weight = data['h2_weights'][weight_idx]
            h2_res[weight_idx] += curr_weight * res

    for idx in range(len(h2_res)):
        h2_res[idx] += data['bias2'][idx]

    for idx in range(len(h2_res)):
        h2_res[idx] = relu(h2_res[idx])
    print(h2_res)
    print(range(len(h2_res)))

    out_res = 10 *[0]
    for output in h2_res:
        for weight_idx in range(out):
            curr_weight = data['out_weights'][weight_idx]
            out_res[weight_idx] += curr_weight * output

    for idx in range(len(out_res)):
        out_res[idx] += data['biasout'][idx]

    for idx in range(len(out_res)):
        out_res[idx] = relu(out_res[idx])
    print(out_res)
    print(range(len(out_res)))



    return


    #with open('C:/Users/bajorat_benjamin/Documents/codenv/mlp_gen/datas.json', 'w') as outfile:
    #    json.dump(weightlist,outfile)


if __name__ == "__main__":
    main()
