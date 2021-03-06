
import datetime

import tensorflow as tf

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential

from tensorflow.keras.activations import tanh

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

from tensorflow.keras.callbacks import TensorBoard

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.expand_dims(x_train, axis = 3)

x_train = tf.image.resize(x_train, (32, 32), method = 'bilinear')

x_train /= 255

y_train = to_categorical(y_train, num_classes = 10)

x_test = tf.expand_dims(x_test, axis = 3)

x_test = tf.image.resize(x_test, (32, 32), method = 'bilinear')

x_test /= 255

y_test = to_categorical(y_test, num_classes = 10)

model = Sequential()

A = 1.7159

Atanh = lambda x: A * tanh(x)

model.add(
        Conv2D(
            6,
            (5, 5),
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last',
            input_shape = (32, 32, 1)))

model.add(
        AveragePooling2D(
            pool_size = (2, 2),
            strides = (2, 2),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(Lambda(Atanh))

model.add(
        Conv2D(
            16,
            (5, 5),
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(
        AveragePooling2D(
            pool_size = (2, 2),
            strides = (2, 2),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(Lambda(Atanh))

model.add(
        Conv2D(
            120,
            (5, 5),
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(Flatten())

model.add(Dense(84))

model.add(Lambda(Atanh))

model.add(Dense(10))

model.add(Activation('softmax'))

model.compile(
        optimizer = 'SGD',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

log_dir = './logs/train/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

tensorboard = TensorBoard(
        log_dir = log_dir,
        histogram_freq = 1,
        write_graph = True,
        write_images = True,
        update_freq = 'batch',
        profile_batch = 0,
        embeddings_freq = 1,
        embeddings_metadata = None)

model.fit(
        x = x_train,
        y = y_train,
        batch_size = 32,
        epochs = 100,
        verbose = 2,
        callbacks = [tensorboard])

score = model.evaluate(
        x = x_test,
        y = y_test,
        batch_size = 32,
        verbose = 2,
        callbacks = [tensorboard],
        return_dict = True)

print(f'Test loss: {score["loss"]}')

print(f'Test accuracy: {score["accuracy"]}')

model.save('model.h5', include_optimizer = False)

