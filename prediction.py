# CS 165B HW4 - Tomas Vera

"""
Implement the testing procedure here. 

Inputs:
    Unzip the hw4_test.zip and place the folder named "hw4_test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        otherwise you will be given 0 score for your hw4.

"""

MODEL = 'model2.h5'
PREDICTION_FILE = 'test.txt'

import os
from PIL import Image

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model


test_folder = "hw4_test"


def load_testset(test_folder):
    test_images=[]

    for i in range(10000):
        print("Loading test image: " + str(i), end = "\r")
        test_image_path= os.path.join(test_folder, str(i)+".png")

        test_image = Image.open(test_image_path).getdata()
        test_image = np.array(test_image).reshape(28, 28)
        test_images.append(test_image)
    
    test_images = np.array(test_images)

    return test_images

print("---Loading images---")
test_images = load_testset(test_folder)
print("Done loading images")

test_images = test_images / 255.0


model = load_model(MODEL)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

with open(PREDICTION_FILE, 'w') as f:
    for i in range(10000):
        f.write(str(np.argmax(predictions[i])))
        f.write('\n')