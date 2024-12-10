import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D ,LeakyReLU
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model   # from tf.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from livelossplot.inputs.keras import PlotLossesCallback
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np


def inception_fine_tune(input_shape, n_classes, x_train, y_train):
    inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)
    for layer in inception_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(x_train, y_train)

    for i, layer in enumerate(model.layers):

        if i < 249:
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(x_train, y_train, batch_size=256, epochs=50, shuffle=True, validation_split=0.1)
    return model   # history


def create_model(input_shape, n_classes, fine_tune=0):
    FREEZE_LAYERS = 249
    # build our classifier model based on pre-trained InceptionResNetV2:
    # 1. we don't include the top (fully connected) layers of InceptionResNetV2
    # 2. we add a DropOut layer followed by a Dense (fully connected)
    #    layer which generates softmax class score for each class
    # 3. we compile the final model using an Adam optimizer, with a low learning rate (since we are 'fine-tuning')
    conv_base = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

    x = conv_base.output
    pooling = GlobalAveragePooling2D()(x)
    out = Dense(1024)(pooling)
    out = LeakyReLU(alpha=0.2)(out)
    out = Dense(n_classes, activation="softmax")(out)
    model = Model(inputs=conv_base.input, outputs=out)

    for layer in conv_base.layers:
        layer.trainable = False

    # # add a global spatial average pooling layer
    # x = conv_base.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(128, activation='relu')(x)
    # # and a logistic layer
    # output_layer = Dense(n_classes, activation='softmax', name='softmax')(x)
    #
    # # this is the model we will train
    # model = Model(inputs=conv_base.input, outputs=output_layer)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in base_model.layers:
    #     layer.trainable = False

    # for layer in conv_base.layers[:FREEZE_LAYERS]:
    #     layer.trainable = False
    # for layer in conv_base.layers[FREEZE_LAYERS:]:
    #     layer.trainable = True
    #
    # x = conv_base.output
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # output_layer = Dense(n_classes, activation='softmax', name='softmax')(x)
    # model = Model(inputs=conv_base.input, outputs=output_layer)

    # SGD(learning_rate=1e-3, momentum=0.9)
    model.compile(optimizer=SGD(learning_rate=1e-3, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    return model


BATCH_SIZE = 64
n_classes = 2
n_epochs = 5
input_shape = (75, 75, 3)

model_save_path = r'E:\senem\chagas_project\classification_inceptionv3\inceptionv3_trained_model'
data = pd.read_csv("./classification_data/inceptionv3_75_normalized.csv")
dict_data = np.load('./classification_data/inceptionv3_75_normalized.npz')

y = data.ClassLabel
dummy_y = np_utils.to_categorical(y.map(dict(si=1, no=0)), dtype="uint8")
X = dict_data['arr_0']   # extract the first array

# PREPROCESSING X = preprocess_input(X, data_format=None)  # mode='torch'
# from keras.applications.inception_v3 import preprocess_input
# X = preprocess_input(X, data_format=None)
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.2, random_state=42)

n_steps = len(X_train) // BATCH_SIZE      # X_train.samples // BATCH_SIZE
n_val_steps = len(X_valid) // BATCH_SIZE

# model = create_model(input_shape, n_classes, fine_tune=20)
model = inception_fine_tune(input_shape, n_classes, X_train, y_train)

# import tensorflow as tf
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

from keras.callbacks import ReduceLROnPlateau
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30)
hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=n_epochs, validation_data=(X_valid, y_valid),
                 steps_per_epoch=n_steps, validation_steps=n_val_steps, callbacks=[rlrop], verbose=1)

model.save(model_save_path)

# ---------------------------------------------------
#                  DRAW THE RESULT
# ---------------------------------------------------
import matplotlib.pyplot as plt  # %matplotlib inline
# plt.style.use('fivethirtyeight')
plt.subplot(211)
print(hist.history.keys())
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.ylabel('accuracy')
plt.legend(['train','test'], loc=3,ncol=6,mode="expand",
           bbox_to_anchor=(0., 1.02, 1., .102), borderaxespad=0.)
plt.subplot(212)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()