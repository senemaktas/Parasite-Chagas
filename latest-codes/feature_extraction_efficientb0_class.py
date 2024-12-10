# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#transfer-learning-from-pretrained-weights
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import Model   # from tf.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from livelossplot.inputs.keras import PlotLossesCallback
from tensorflow.keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
# from keras.applications.vgg16 import preprocess_input as keras_preprocess
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import keras

# def unfreeze_model(model):
#     # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
#     for layer in model.layers[-5:]:
#         if not isinstance(layer, layers.BatchNormalization):
#             layer.trainable = True
#
#     optimizer = Adam(learning_rate=1e-3)
#     model.compile(
#         optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
#     )
#
#     return model


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    # optimizer = Adam(learning_rate=0.0001)
    optimizer = SGD(learning_rate=0.001, momentum=0.9, decay=0.1)  # , decay=0.5 , momentum=0.9, decay=0.1
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


BATCH_SIZE = 64
IMG_SIZE = 224
NUM_CLASSES = 2
epochs = 25

# READ THE DATASET
data = pd.read_csv("../classification_folder/chagas_class_csv.csv")
data.head()
data.isnull().values.any()  # check if the dataset contains any NULL value or not.

y = data.ClassLabel
dummy_y = np_utils.to_categorical(y.map(dict(si=1, no=0)), dtype="uint8")

dict_data = np.load('../classification_folder/compressed_chagas_images.npz')
X = dict_data['arr_0']   # extract the first array

# PREPROCESSING
X = preprocess_input(X, data_format=None)  # mode='torch'

X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.2, random_state=42)

model = build_model(num_classes=NUM_CLASSES)
# model = unfreeze_model(model)

n_steps = len(X_train) // BATCH_SIZE      # X_train.samples // BATCH_SIZE
n_val_steps = len(X_valid) // BATCH_SIZE

import tensorflow as tf
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-2 * 10**(epoch/20))

from keras.callbacks import ReduceLROnPlateau
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30)

hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_data=(X_valid, y_valid),
                 steps_per_epoch=n_steps, validation_steps=n_val_steps,
                 callbacks=[lr_scheduler, rlrop], verbose=1)
# plot_hist(hist)

# save the model chagas_classification_model_cpu_sgd_sigmoid_binary_epoch15
chagas_classification_model_path = r'E:\senem\chagas_project\classification_efficient\efficient_gray_cpu_sgd_001_softmax_binary_epoch15'
model.save(chagas_classification_model_path)

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
