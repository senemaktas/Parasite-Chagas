import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
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


def create_model(input_shape, n_classes, fine_tune=0):
    FREEZE_LAYERS = 2
    # build our classifier model based on pre-trained InceptionResNetV2:
    # 1. we don't include the top (fully connected) layers of InceptionResNetV2
    # 2. we add a DropOut layer followed by a Dense (fully connected)
    #    layer which generates softmax class score for each class
    # 3. we compile the final model using an Adam optimizer, with a low learning rate (since we are 'fine-tuning')
    conv_base = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in conv_base.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in conv_base.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    x = conv_base.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(n_classes, activation='softmax', name='softmax')(x)
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # SGD(learning_rate=1e-3, momentum=0.9)
    model.compile(optimizer=SGD(learning_rate=1e-3, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    return model


BATCH_SIZE = 64
n_classes = 2
n_epochs = 15
input_shape = (75, 75, 3)

model_save_path = r'E:\senem\chagas_project\classification_inceptionresnetv2\inceptionresnetv2_trained_model'
data = pd.read_csv("./classification_data/inceptionresnetv2_75_normalized.csv")
dict_data = np.load('./classification_data/inceptionresnetv2_75_normalized.npz')

y = data.ClassLabel
dummy_y = np_utils.to_categorical(y.map(dict(si=1, no=0)), dtype="uint8")
X = dict_data['arr_0']   # extract the first array

# PREPROCESSING X = preprocess_input(X, data_format=None)  # mode='torch'
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.2, random_state=42)

n_steps = len(X_train) // BATCH_SIZE      # X_train.samples // BATCH_SIZE
n_val_steps = len(X_valid) // BATCH_SIZE

model = create_model(input_shape, n_classes, fine_tune=20)

# import tensorflow as tf
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

from keras.callbacks import ReduceLROnPlateau
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30)
hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=n_epochs, validation_data=(X_valid, y_valid),
                 steps_per_epoch=n_steps, validation_steps=n_val_steps, callbacks=[rlrop], verbose=1)

# hist = model.fit_generator((X_train, y_train),
#                            steps_per_epoch=n_steps,  # train_batches.samples // BATCH_SIZE,
#                            validation_data=(X_valid, y_valid),
#                            validation_steps=n_val_steps, # valid_batches.samples // BATCH_SIZE,
#                            epochs=n_epochs)

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