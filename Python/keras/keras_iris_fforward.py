from keras.models import model_from_json
import numpy as np

test_x = np.array([5.1, 3.3, 1.7, 0.5])
x = test_x.reshape((1,4))
test_y =[1,2,0]

# load json and create model
json_file = open('C:/Users/Ben/Desktop/ann_data/Keras_iris/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
keras_model = model_from_json(loaded_model_json)
# load weights into new model
keras_model.load_weights('C:/Users/Ben/Desktop/ann_data/Keras_iris/model.h5')
#keras_model._make_predict_function()


pred = keras_model.predict(x, batch_size=1)
for array in pred:
        for elem in array:
                print(elem)
