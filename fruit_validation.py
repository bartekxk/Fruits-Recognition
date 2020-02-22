import tensorflow as tf
import tensorboard
import numpy as np
import matplotlib.pyplot as plt
import glob

path = 'fruits-360_dataset/fruits-360/Test/Apple Golden 3/31_100.jpg'

directory = './fruits-360_dataset/fruits-360/Training'
classes = np.array([w[41:] for w in glob.glob(directory + '/*')])
img = tf.keras.preprocessing.image.load_img(path=path, target_size=(100,100))
inputJpg = tf.keras.preprocessing.image.img_to_array(img)
inputJpg = inputJpg[np.newaxis, ...]

model = tf.keras.models.load_model('f_model2.h5', compile=False)
result = model.predict(inputJpg)
resultClass = np.where(result == result.max())
print(classes[resultClass[1][0]])