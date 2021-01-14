from __future__ import print_function
import h5py
import json
import numpy as np

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items()) == 0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()


def weight_to_list(double_array, new_list):
    counter = 0
    for arrays in double_array:
        for array in arrays:
            for elem in array:
                new_list.append(elem)
                counter = counter + 1
    print("total_weights: {}".format(counter))
    return new_list


def bias_to_list(array, new_list):
    counter = 0
    for elem in array:
        for weight in elem:
            new_list.append(weight)
            counter = counter + 1
    print("total_biases: {}".format(counter))
    return new_list

def main():

    #print_structure('C:/Users/Ben/Desktop/ann_data/Keras_mnist/model.h5')

    b1_weights, b2_weights, b3_weights, h1_weights, h2_weights, h3_weights = [], [], [], [], [], []
    f = h5py.File('C:/Users/Ben/Desktop/ann_data/Keras_mnist/model.h5')
    for layer, g in f.items():
        for p_name in g.keys():
            param = g[p_name]
            for k_name in param.keys():
                # weightlist.append(param.get(k_name)[:])
                if (p_name == "dense_1" and k_name == "bias:0"):
                    b1_weights.append(param.get(k_name)[:])
                if (p_name == "dense_1" and k_name == "kernel:0"):
                    h1_weights.append(param.get(k_name)[:])
                if (p_name == "dense_2" and k_name == "bias:0"):
                    b2_weights.append(param.get(k_name)[:])
                if (p_name == "dense_2" and k_name == "kernel:0"):
                    h2_weights.append(param.get(k_name)[:])
                if (p_name == "dense_3" and k_name == "bias:0"):
                    b3_weights.append(param.get(k_name)[:])
                if (p_name == "dense_3" and k_name == "kernel:0"):
                    h3_weights.append(param.get(k_name)[:])

    h1, h2, h3, b1, b2, b3 = [], [], [], [], [], []
    weight_to_list(h1_weights, h1)
    bias_to_list(b1_weights, b1)
    weight_to_list(h2_weights, h2)
    bias_to_list(b2_weights, b2)
    weight_to_list(h3_weights, h3)
    bias_to_list(b3_weights, b3)

    # cumulated list
    weightlist = h1 + b1 + h2 + b2 + h3 + b3

    weights = []
    for weight in weightlist:
        weights.append(float(weight))

    count_w = 0
    for i in weightlist:
        count_w = count_w + 1

    # export weights to JSON
    with open('C:/Users/Ben/Desktop/ann_data/Keras_mnist/mnist_weights.json', 'w') as outfile:
        json.dump(weights, outfile)
    print("number of weights total: ", count_w)


if __name__ == "__main__":
    main()
