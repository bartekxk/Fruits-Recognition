# Fruits-Recognition
Fruits Recognition using tensorflow and keras libraries. Dataset was taken from kaggle website. With validation.py script we can upload any fruit image and recognize it.

## Model 
    tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=(2, 2), activation='relu', input_shape=(100,100,3)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D((3,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(120, activation='softmax')
