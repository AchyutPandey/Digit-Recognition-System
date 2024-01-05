
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


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=15)

loss,accuracy=model.evaluate(X_test,Y_test)

"""Accuracy is 97.28"""

plt.imshow(X_test[0])
plt.show()
print(Y_test[0])

Y_pred=model.predict(X_test)

print(Y_pred.shape)

Y_pred_labels=[np.argmax(i) for i in Y_pred]

print(Y_pred_labels)

conf_mat=confusion_matrix(Y_test,Y_pred_labels)

print(conf_mat)

plt.figure(figsize=(15,7))
sns.heatmap(conf_mat,annot=True,fmt='d',cmap='Blues')
plt.ylabel('Truelabel')
plt.xlabel('Predicted Labels')

"""Predictive System"""

input_image_path=input('Path of image to be predicted: ')
input_image=cv2.imread(input_image_path)

grayscale=cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)

input_image_resize=cv2.resize(grayscale,(28,28))
cv2_imshow(input_image)
input_image_resize=input_image_resize/255
image_reshaped=np.reshape(input_image_resize,[1,28,28])
input_prediction=model.predict(image_reshaped)
input_pred_label=np.argmax(input_prediction)
print("The digit is recognised as: ",input_pred_label)
