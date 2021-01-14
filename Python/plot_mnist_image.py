from tensorflow.examples.tutorials.mnist import input_data
#import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)    
    
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.float32)#(np.uint8), (np.bool), (np.byte), (np.float), (np.int)
    twod = np.where(two_d > 0.5, 1, 0)
    print(twod)
    plt.imshow(twod)# ,interpolation='nearest')
    return plt

# Get a batch of two random images and show in a pop-up window.
batch_xs, batch_ys = mnist.test.next_batch(2)
gen_image(batch_xs[0]).show()
gen_image(batch_xs[1]).show()


batch_xs, batch_ys = mnist.train.next_batch(2)
gen_image(batch_xs[0]).show()
gen_image(batch_xs[1]).show()
