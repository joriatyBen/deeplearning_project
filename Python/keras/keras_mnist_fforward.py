from keras.models import model_from_json
import numpy as np
from math import exp
import json
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# load mnist data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
#mnist = input_data.read_data_sets("P:/mlp_test/Playground/mnist_c_test/data", one_hot=True)

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

batch_xs, batch_ys = mnist.train.next_batch(2)
#two_as_array = gen_image(batch_xs[0])
two_vec = batch_xs[0]
print(two_vec)
print("label: ",batch_ys[0])

# config mnist picture in bit
#reshape_vec = (np.reshape(two_vec, (28, 28)) * 255).astype(np.float32)
#reshape_vec = np.where(reshape_vec >0, 1, 0)
#print(reshape_vec)

x = two_vec.reshape((1,784))

# load json and create model
json_file = open('C:/Users/Ben/Desktop/ann_data/Keras_mnist/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
keras_model = model_from_json(loaded_model_json)
# load weights into new model
keras_model.load_weights('C:/Users/Ben/Desktop/ann_data/Keras_mnist/model.h5')
#keras_model._make_predict_function()


pred = keras_model.predict(x, batch_size=1)
elements=[]
for array in pred:
        for elem, idx in zip(array, range(len(array))):
            print(elem)
            elements.append(elem)
print("max:", np.amax(elements))
