# https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/
import os
# # error :  Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. icinn
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'True'
# # comment out below line to enable tensorflow logging outputs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# gpus = tf.config.experimental.list_physical_devices('GPU')
# visible_devices = tf.config.get_visible_devices()
# for devices in visible_devices:
#     print(devices)
# from tensorflow.keras import mixed_precision
# # policy = mixed_precision.Policy('mixed_float16')
# # mixed_precision.set_global_policy(policy)
# # Equivalent to the two lines above
# mixed_precision.set_global_policy('mixed_float16')

# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
from tensorflow.keras.models import Model   # from tf.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from livelossplot.inputs.keras import PlotLossesCallback
from tensorflow.keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import pandas as pd
import numpy as np

BATCH_SIZE = 64

# train_generator = ImageDataGenerator(rotation_range=90, brightness_range=[0.1, 0.7], width_shift_range=0.5,
#                                      height_shift_range=0.5, horizontal_flip=True, vertical_flip=True,
#                                      validation_split=0.15, preprocessing_function=preprocess_input)
#
# test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
#
# download_dir = Path('./daaaattaa/')
#
# train_data_dir = download_dir/'food-101/train'
# test_data_dir = download_dir/'food-101/test'
#
# class_subset = sorted(os.listdir(download_dir/'food-101/images'))[:10]  # Using only the first 10 classes
#
# traingen = train_generator.flow_from_directory(train_data_dir, target_size=(224, 224), class_mode='categorical',
#                                                classes=class_subset, subset='training', batch_size=BATCH_SIZE,
#                                                shuffle=True, seed=42)
#
# validgen = train_generator.flow_from_directory(train_data_dir, target_size=(224, 224), class_mode='categorical',
#                                                classes=class_subset,  subset='validation', batch_size=BATCH_SIZE,
#                                                shuffle=True, seed=42)
#
# testgen = test_generator.flow_from_directory(test_data_dir, target_size=(224, 224), class_mode=None,
#                                              classes=class_subset, batch_size=1, shuffle=False, seed=42)
# ---------------------------------------------------
#                  READ THE DATASET
# ---------------------------------------------------
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input as keras_preprocess
data = pd.read_csv("chagas_class_gray.csv")
data.head()
data.isnull().values.any()  # check if the dataset contains any NULL value or not.

y = data.ClassLabel
dummy_y = np_utils.to_categorical(y.map(dict(si=1, no=0)), dtype="uint8")

dict_data = np.load('compressed_chagas_class_gray.npz')
X = dict_data['arr_0']   # extract the first array

# ---------------------------------------------------
#                  PREPROCESSING
# ---------------------------------------------------
X = keras_preprocess(X, data_format=None)  # mode='torch'

X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.2, random_state=42)


def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pretrained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers. 'imagenet'
    conv_base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(128, activation='relu')(top_model)
    top_model = Dense(64, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)  # softmax , tanh, sigmoid

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    # adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    sgd = SGD(learning_rate=0.01, momentum=0.9, decay=0.01)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])  # mse, mea, binary_crossentropy

    return model


# -----------------------------------------------
# Using Pre-trained Layers for Fine-Tuning
# -----------------------------------------------
# Reset our image data generators traingen.reset() validgen.reset() testgen.reset()
n_classes = 2
n_epochs = 15
input_shape = (70, 70, 3)
optim_2 = SGD(learning_rate=0.001)  # Adam(learning_rate=0.0001)  # Use a smaller learning rate
n_steps = len(X_train) // BATCH_SIZE      # X_train.samples // BATCH_SIZE
n_val_steps = len(X_valid) // BATCH_SIZE

# Re-compile the model, this time leaving the last 2 layers unfrozen for Fine-Tuning
vgg_model_ft = create_model(input_shape, n_classes, optim_2, fine_tune=2)

# plot_loss_2 = PlotLossesCallback()  # for notebooks callbacks=[tl_checkpoint_1, early_stop, plot_loss_2]
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5', save_best_only=True, verbose=1)

from keras.utils.vis_utils import plot_model
plot_model(vgg_model_ft, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Retrain model with fine-tuning
from keras.callbacks import ReduceLROnPlateau
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30)
vgg_ft_history = vgg_model_ft.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=n_epochs, validation_data=(X_valid, y_valid),
                                  steps_per_epoch=n_steps, validation_steps=n_val_steps,
                                  callbacks=[rlrop, tl_checkpoint_1, early_stop], verbose=1)

# save the model chagas_classification_model_cpu_sgd_sigmoid_binary_epoch15
chagas_classification_model_path = r'E:\senem\chagas_project\classification_folder\gray_cpu_sgd_001_sigmoid_binary_epoch15'
vgg_model_ft.save(chagas_classification_model_path)
# ---------------------------------------------------
#                  DRAW THE RESULT
# ---------------------------------------------------
import matplotlib.pyplot as plt  # %matplotlib inline
plt.subplot(211)
print(vgg_ft_history.history.keys())
plt.plot(vgg_ft_history.history['acc'])
plt.plot(vgg_ft_history.history['val_acc'])
plt.ylabel('accuracy')
plt.legend(['train','test'], loc=3,ncol=6,mode="expand",
           bbox_to_anchor=(0., 1.02, 1., .102), borderaxespad=0.)
plt.subplot(212)
plt.plot(vgg_ft_history.history['loss'])
plt.plot(vgg_ft_history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

