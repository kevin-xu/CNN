
import numpy as np

import scipy as sp

import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow.keras.activations import tanh

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = load_model('model.h5', custom_objects = {'A': 1.7159, 'tanh': tanh})

x = tf.io.read_file('x.jpg')

x = tf.image.decode_jpeg(x, channels = 1)

x = x.numpy()

x = sp.ndimage.zoom(x, (32 / x.shape[0], 32 / x.shape[1], 1), order = 1)

x = x / 255

x = np.expand_dims(x, axis = 0)

y = model.predict(x, verbose = 1)

print(y)

print(np.argmax(y))

