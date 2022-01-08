import os

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

np.random.seed(1337)



def load_DRModel(kernel_size, nb_filters, channels, img_rows, img_cols, nb_classes):
    """
    Define and run the Convolutional Neural Network

    INPUT
        X_train: Array of NumPy arrays
        X_test: Array of NumPy arrays
        y_train: Array of labels
        y_test: Array of labels
        kernel_size: Initial size of kernel
        nb_filters: Initial number of filters
        channels: Specify if the image is grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification

    OUTPUT
        Fitted CNN model
    """
    
	dr_weights = 'models/retinopathy_melanoma_DR_Two_Classes_recall_0.0.h5'
    
	model = Sequential()

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     strides=1,
                     input_shape=(img_rows, img_cols, channels), activation="relu"))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)

    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.load_weights(dr_weights)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



if __name__ == '__main__':
   
    # Specify parameters before model is run.
#   batch_size = 1000
    batch_size = 10
    nb_classes = 2
    nb_epoch = 30

    img_rows, img_cols = 256, 256
    channels = 3
    nb_filters = 32
    kernel_size = (8, 8)   

    img_path = 'data/cnn_data/balanced/test/processed/malignant/ISIC_0012086.jpg'
	    
	img = cv2.resize(cv2.imread(img_path), (img_width, img_height))     # load image and resize
	
    img = img.transpose((2,0,1))    # rearrange image channels for correctly formatted input into the Keras model
    img = np.expand_dims(img, axis=0)

	print("Loading Model")
	
    # Load pretrained model
    model = load_DRModel(kernel_size, nb_filters, channels, img_rows, img_cols, nb_classes)
    
	out = model.predict(img)
    
    class_predicted = model.predict_classes(out)

    print(" Class Predicted: ", class_predicted, '\n')



    input_shape = (img_rows, img_cols, channels)

    print("Normalizing Data")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

 


    print("Predicting")
    y_pred = model.predict(X_test)

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    y_pred = [np.argmax(y) for y in y_pred]
    y_test = [np.argmax(y) for y in y_test]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Precision: ", precision)
    print("Recall: ", recall)

    save_model(model=model, score=recall, model_name="DR_Two_Classes")
    print("Completed")
