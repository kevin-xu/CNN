
import sys

import datetime

import numpy as np

import scipy as sp

import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Layer
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
    return tf.cast(example['image'], dtype = 'float32'), example['label']

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
        )

def reduce_sum(old_sum_rgb, xy):
    x, _ = xy

    x = tf.reshape(x, (-1, 3))

    return old_sum_rgb + tf.reduce_sum(x, axis = 0)

sum_rgb = ds_train.reduce(tf.zeros((3,)), reduce_sum)

mean_activity = sum_rgb / ds_train_size

def subtract_mean_activity(xy):
    x, y = xy

    return x - mean_activity, y

ds_train = (
        ds_train
        .map(subtract_mean_activity, num_parallel_calls = 8)
        .interleave(crop_to_224x224_and_flip, num_parallel_calls = 8)
        .prefetch(1024)
        )

def reshape_and_concat(old_X, xy):
    x, _ = xy

    x = tf.reshape(x, (-1, 3))

    return tf.concat((old_X, x), 0)

X = ds_train.reduce(tf.zeros((0, 3)), reshape_and_concat)

X = tf.transpose(X)

s, U, _ = tf.linalg.svd(X, full_matrices = True)

def TrainingGenerator(ds, ds_size, batch_size, n_epochs):
    for _ in range(n_epochs):
        _ds = ds.shuffle(ds_size)

        batched_ds = _ds.batch(batch_size)

        for i, batch in enumerate(batched_ds):
            X = tf.zeros((batch_size, 224, 224, 3))

            Y = tf.zeros((batch_size,), dtype = 'int64')

            for xy in batch:
                x, y = xy

                alpha = tf.random.normal((3, 1), mean = 0.0, stddev = 0.1)

                delta = tf.matmul(U, alpha * s)

                x += tf.transpose(delta)

                X[i] = x

                Y[i] = y

            Y = to_categorical(Y, num_classes = 1000)

            yield X, Y

ds_test = (
        ds['test']
        .map(to_xy, num_parallel_calls = 8)
        .map(resize_and_crop_to_256x256, num_parallel_calls = 8)
        .map(subtract_mean_activity, num_parallel_calls = 8)
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

        x = tf.cast(x, dtype = 'float32')

        x -= mean_activity

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

class LocalResponseNormalization(Layer):
    def __init__(
            self,
            depth_radius = 5,
            bias = 1.0,
            alpha = 1.0,
            beta = 0.5,
            name = None,
            **kwargs):
        super().__init__(self, trainable = False, name = name, **kwargs)

        self._depth_radius = depth_radius

        self._bias = bias

        self._alpha = alpha

        self._beta = beta

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, tf.Tensor):
            raise TypeError

        input_ = inputs

        output = tf.nn.local_response_normalization(
                input_,
                depth_radius = self._depth_radius,
                bias = self._bias,
                alpha = self._alpha,
                beta = self._beta,
                name = self.name)

        return output

model.add(
        LocalResponseNormalization(
            depth_radius = 5,
            bias = 2.0,
            alpha = 1.0e-4,
            beta = 0.75))

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
        LocalResponseNormalization(
            depth_radius = 5,
            bias = 2.0,
            alpha = 1.0e-4,
            beta = 0.75))

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

model.add(Dense(4096))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(4096))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1000))

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

training_batch_size = 128

training_n_epochs = 10

model.fit(
        x = TrainingGenerator(
            ds_train,
            ds_train_size,
            training_batch_size,
            training_n_epochs),
        epochs = training_n_epochs,
        steps_per_epoch = ds_train_size / training_batch_size,
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

