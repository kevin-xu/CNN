
import tensorflow as tf

from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = load_model('model.tf')

assert tf.io.gfile.isdir('xs')

walk = tf.io.gfile.walk('xs')

_, _, xns = next(walk)

xs = []

for x in xns:
    x = tf.io.read_file('xs/' + x)

    x = tf.image.decode_image(x, channels = 3)

    xs.append(x)

xs = tf.image.resize(xs, (256, 256), method = 'bilinear')

ys = model.predict(xs, verbose = 1)

print(ys)

print(tf.argmax(ys, axis = 1))

