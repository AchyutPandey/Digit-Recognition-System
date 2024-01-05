
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

#shape
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

"""60000 images of dimension 28X28"""

print(X_train[1].shape)

plt.imshow(X_train[40])
plt.show()
print(Y_train[40])

print(np.unique(Y_train))
print(np.unique(Y_test))

X_train=X_train/255
X_test=X_test/255

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(50,activation="relu"),
    keras.layers.Dense(50,activation="relu"),
    keras.layers.Dense(10,activation="sigmoid")
])
