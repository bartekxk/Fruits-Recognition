import tensorflow as tf
import tensorboard
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

alpha = 0.02
iterations = 50
batch_size = 32
display_step = 5
img_width = 100
img_height = 100

# def dataLoader(loadType):
#     image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
#     directory = './fruits-360_dataset/fruits-360/' + loadType
#     classes = np.array([w[41:] for w in glob.glob(directory + '/*')])
#     data_set = image_generator.flow_from_directory(directory=directory,
#                                                    batch_size=batch_size,
#                                                    shuffle=True,
#                                                    target_size=(img_height, img_width),
#                                                    classes=list(classes))
#     return data_set, classes

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
trainingDir = './fruits-360_dataset/fruits-360/Training'
testDir = './fruits-360_dataset/fruits-360/Test'
classes = np.array([w[41:] for w in glob.glob(trainingDir + '/*')])

def show_batch(image_batch, label_batch, class_names):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(class_names[label_batch[n]==1][0].title())
      plt.axis('off')


#data_set, classes = dataLoader(loadType='Training')
#image_batch, label_batch = next(data_set)
#show_batch(image_batch, label_batch, classes)
#plt.show()

logdir = "tmp/logs/" +datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=(2, 2), activation='relu', input_shape=(100,100,3)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D((3,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(120, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

traning_history = model.fit(image_generator.flow_from_directory(directory=trainingDir,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   target_size=(img_height, img_width),
                                                   classes=list(classes)),
                                                   batch_size=batch_size, epochs = 5,
                                                   workers=4)
model.save('f_model2.h5')
model.evaluate(image_generator.flow_from_directory(directory=testDir,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   target_size=(img_height, img_width),
                                                   classes=list(classes)),
                                                   verbose = 2)