
import numpy as np

import scipy as sp

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model('model.h5')

x = load_img('digit.jpg', color_mode = 'grayscale')

x = img_to_array(x)

x = sp.ndimage.zoom(x, (32 / x.shape[0], 32 / x.shape[1], 1), order = 1)

x = x / 255

x = np.expand_dims(x, axis = 0)

y = model.predict(x, verbose = 1)

print(y)

print(np.argmax(y))

