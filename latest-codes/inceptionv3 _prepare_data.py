import os
# error :  Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. icinn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'True'
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import csv
# import tensorflow as tf
from keras.utils import np_utils
import pandas as pd
import numpy as np
from numpy import savez_compressed

# ---------------------------------------
# prepare cvs file
# ---------------------------------------
# from keras.applications.inception_v3 import preprocess_input
resize_shape = 75
image_folder = '../chagas-capilares/imagenes/gris/'
csv_path = './classification_data/inceptionv3_75_normalized.csv'
npz_path = './classification_data/inceptionv3_75_normalized.npz'

from pathlib import Path
Path("./classification_data").mkdir(parents=True, exist_ok=True)


with open(csv_path, 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    csv_writer.writerow(['image_full_path', 'ClassLabel'])

    for inner_dir in os.listdir(image_folder):   # si and no label
        for filename in os.listdir(image_folder + inner_dir):
            # name, ext = os.path.splitext(os.path.basename(filename))
            image_full_path = image_folder + "/" + inner_dir + "/" + filename
            csv_writer.writerow([str(image_full_path), inner_dir])

data = pd.read_csv(csv_path)

X = []
for row in data.image_full_path:
    row = ''.join(str(row) for row in row)
    X.append(row)
X = np.array(X)

y = data.ClassLabel
dummy_y = np_utils.to_categorical(y.map(dict(si=1, no=0)), dtype="uint8")

images = []
for i in range(0, X.shape[0]):
    # b = cv2.fastNlMeansDenoisingColored(cv2.imread(X[i]), None, 10, 10, 7, 21)
    # a = resize(b, preserve_range=True,output_shape=(224, 224, 3)).astype(int)
    a = cv2.resize(np.asarray(cv2.imread(X[i])), (resize_shape, resize_shape), interpolation=cv2.INTER_AREA)
    grayscale_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    a = cv2.merge((grayscale_a, grayscale_a, grayscale_a))
    # a = preprocess_input(a)
    a = np.array(a) / 255.0
    # normalizedData = (a - np.min(a)) / (np.max(a) - np.min(a))
    images.append(a)
X = np.array(images)

# önişlemde sonra tekrardan kullanmak için verilerin arraye depolanması işlemi.
savez_compressed(npz_path, X)



