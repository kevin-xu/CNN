
import sys

import datetime

import numpy as np

import scipy as sp

import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow.keras.utils import Sequence

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras import Model

imagenet2012_path = sys.argv[1]

ds, ds_info = tfds.load(
        'Imagenet2012',
        shuffle_files = True,
        data_dir = imagenet2012_path,
        with_info = True)

def to_xy(example):
    return example['image'], example['label']

def resize_and_crop_to_256x256(xy):
    x, y = xy

    h, w = x.shape[1], x.shape[2]

    if h < w:
        _w = 256 // h * w
        if _w % 2 != 0:
            _w += 1

        x = tf.image.resize(x, (256, _w), method = 'bilinear')

        x = tf.image.crop_to_bounding_box(x, 0, (_w - 256) // 2, 256, 256)

    elif h == w:
        x = tf.image.resize(x, (256, 256), method = 'bilinear')

    else:
        _h = 256 // w * h
        if _h % 2 != 0:
            _h += 1

        x = tf.image.resize(x, (_h, 256), method = 'bilinear')

        x = tf.image.crop_to_bounding_box(x, (_h - 256) // 2, 0, 256, 256)

    return x, y

def crop_to_224x224_and_flip(xy):
    x, y = xy

    xs = []

    for i in range(32):
        for j in range(32):
            _x = tf.image.crop_to_bounding_box(x, i, j, 224, 224)

            fx = tf.image.flip_left_right(_x)

            xs.extend((_x, fx))

    return tf.data.Dataset.from_tensor_slices((xs, [y for i in range(2048)]))

ds_train_size = ds_info.splits['train'].num_examples * 2048

ds_train = (
        ds['train']
        .map(to_xy, num_parallel_calls = 8)
        .map(resize_and_crop_to_256x256, num_parallel_calls = 8)
        .interleave(crop_to_224x224_and_flip, num_parallel_calls = 8)
        .shuffle(ds_train_size)
        .batch(256)
        .prefetch(1024)
        )

class XYGenerator(Sequence):
    def __init__(self, ds, ds_size):
        super().__init__()

        self._ds = ds

        self._ds_size = ds_size

    def __len__(self):
        return self._ds_size

    def __getitem__(self, index):
        pass

training_generator = XYGenerator(ds_train, ds_train_size)

ds_test = (
        ds['test']
        .map(to_xy, num_parallel_calls = 8)
        .map(resize_and_crop_to_256x256, num_parallel_calls = 8)
        .shuffle(ds_info.splits['test'].num_examples)
        .batch(256)
        .prefetch(1024)
        )

class AlexNet(Sequential):
    def _predict(self, x):
        x0 = tf.image.crop_to_bounding_box(x, 16, 16, 224, 224)
        x1 = tf.image.crop_to_bounding_box(x, 0, 0, 224, 224)
        x2 = tf.image.crop_to_bounding_box(x, 0, 32, 224, 224)
        x3 = tf.image.crop_to_bounding_box(x, 32, 32, 224, 224)
        x4 = tf.image.crop_to_bounding_box(x, 32, 0, 224, 224)

        fx0 = tf.image.flip_left_right(x0)
        fx1 = tf.image.flip_left_right(x1)
        fx2 = tf.image.flip_left_right(x2)
        fx3 = tf.image.flip_left_right(x3)
        fx4 = tf.image.flip_left_right(x4)

        y0 = self(x0, training = False)
        y1 = self(x1, training = False)
        y2 = self(x2, training = False)
        y3 = self(x3, training = False)
        y4 = self(x4, training = False)

        fy0 = self(fx0, training = False)
        fy1 = self(fx1, training = False)
        fy2 = self(fx2, training = False)
        fy3 = self(fx3, training = False)
        fy4 = self(fx4, training = False)

        return (y0 + y1 + y2 + y3 + y4 + fy0 + fy1 + fy2 + fy3 + fy4) / 10

    def test_step(self, data):
        x, y = data

        _y = self._predict(x)

        self.compiled_loss(y, _y, regularization_losses = self.losses)

        self.compiled_metrics.update_state(y, _y)

        return {metric.name: metric.result() for metric in self.metrics}

    def predict_step(self, data):
        x, _ = data

        return self._predict(x)

model = AlexNet()

model.add(
        Conv2D(
            96,
            (11, 11),
            strides = (4, 4),
            padding = 'valid',
            data_format = 'channels_last',
            input_shape = (224, 224, 3)))

model.add(Activation('relu'))

model.add(
        BatchNormalization(
            axis = -1,
            momentum = 0.99,
            epsilon = 0.001,
            center = True,
            scale = True))

model.add(
        MaxPooling2D(
            pool_size = (3, 3),
            strides = (2, 2),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(
        Conv2D(
            256,
            (5, 5),
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(Activation('relu'))

model.add(
        BatchNormalization(
            axis = -1,
            momentum = 0.99,
            epsilon = 0.001,
            center = True,
            scale = True))

model.add(
        MaxPooling2D(
            pool_size = (3, 3),
            strides = (2, 2),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(
        Conv2D(
            384,
            (3, 3),
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(Activation('relu'))

model.add(
        Conv2D(
            384,
            (3, 3),
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(Activation('relu'))

model.add(
        Conv2D(
            256,
            (3, 3),
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(Activation('relu'))

model.add(
        MaxPooling2D(
            pool_size = (3, 3),
            strides = (2, 2),
            padding = 'valid',
            data_format = 'channels_last'))

model.add(Flatten())

model.add(Dense('4096'))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense('4096'))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense('1000'))

model.add(Activation('softmax'))

model.compile(
        optimizer = 'SGD',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

log_dir = './logs/train/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

tensorboard_callback = TensorBoard(
        log_dir = log_dir,
        histogram_freq = 1,
        write_graph = True,
        write_images = True,
        embeddings_freq = 1,
        embeddings_layer_names = None,
        embeddings_metadata = None,
        embeddings_data = None,
        update_freq = 'batch')

model.fit(
        x = training_generator,
        batch_size = 128,
        epochs = 10,
        verbose = 2,
        callbacks = [tensorboard_callback],
        max_queue_size = 10,
        workers = 2,
        use_multiprocessing = True)

score = model.evaluate(
        x = ds_test,
        batch_size = 128,
        verbose = 2,
        callbacks = [tensorboard_callback],
        max_queue_size = 10,
        workers = 2,
        use_multiprocessing = True,
        return_dict = True)

print(f'Test loss: {score["loss"]}')

print(f'Test accuracy: {score["accuracy"]}')

model.save('model.tf', include_optimizer = False)

