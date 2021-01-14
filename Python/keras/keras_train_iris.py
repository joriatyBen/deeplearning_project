import six.moves.urllib.request as request
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, ReLU
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

# parameter for downloading iris dataset
PATH = "C:/Users/Ben/Desktop/ann_data/Keras_iris/dataset/"
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

# download dataset if not stored yet
#download_dataset(URL_TRAIN, FILE_TRAIN)
#download_dataset(URL_TEST, FILE_TEST)

# locate train/test data
train_dataset = pd.read_csv('C:/Users/Ben/Desktop/ann_data/Keras_iris/dataset/iris_training.csv')
test_dataset = pd.read_csv('C:/Users/Ben/Desktop/ann_data/Keras_iris/dataset/iris_test.csv')

# arrange train data
x_train = train_dataset.iloc[:,0:4].values
y_train = train_dataset.iloc[:,4].values
encoder =  LabelEncoder()
ytr = encoder.fit_transform(y_train)
y_train = pd.get_dummies(ytr).values

# arrange test data
x_test = test_dataset.iloc[:,0:4].values
y_test = test_dataset.iloc[:,4].values
encoder =  LabelEncoder()
yte = encoder.fit_transform(y_test)
y_test = pd.get_dummies(yte).values

# construct multilayer perceptron
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss='mean_squared_error',
              optimizer='adam')

# train model and evaluate with test data
model.fit(x_train, y_train,
                    batch_size=10,
                    epochs=10,
                    verbose=1, #shwos progress_bar
                    shuffle=True,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score)

# serialize model to JSON
model_json = model.to_json()
with open('C:/Users/Ben/Desktop/ann_data/Keras_iris//model.json', 'w') as json_file:
    json_file.write(model_json)
# serialize model to HDF5
model.save_weights('C:/Users/Ben/Desktop/ann_data/Keras_iris/model.h5')