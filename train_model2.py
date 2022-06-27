import pandas as pd
import numpy as np
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import numpy as np


training_folder = "./hw4_train"
test_folder = "hw4_test"

def load_trainset(train_folder):
   
    train_images=[]
    train_labels=[]
   
    for label_folder in os.listdir(train_folder):
        for img_file in os.listdir(os.path.join(train_folder, label_folder)):
       
            image_path= os.path.join(train_folder, label_folder,  img_file)

            image= Image.open(image_path).getdata()
            image=np.array(image).reshape(28, 28)
            train_images.append(image)
            train_labels.append(int(label_folder))
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    return train_images, train_labels

print("---Loading images---")
train_images, train_labels = load_trainset(training_folder)
print("Done loading images")

train_images = train_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28,1)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=100, epochs=7)

model.save('model2.h5')
