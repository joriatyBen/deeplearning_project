from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

test_image=[]
for img in mnist.test.images:
    #img = (np.reshape(img, (28, 28)) * 255).astype(np.float32)
    img = (np.reshape(img, (784)) * 255).astype(np.float32)
    image = np.where(img > 0, 1, 0)
    #print(image)
    #for i in image:
    test_image.append(image)
    for j in test_image:
        #print(range(len(j)))
        print(j)

    #test_image = [[i for i in img] for j in (range(len(mnist.test.images)))]
    #for im in image:
    #    for i in im:
            #print(i)
    #       test_image.append(i)



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


