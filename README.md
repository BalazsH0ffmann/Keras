# Keras
Keras using through clothes
#import libraries#
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Creating the dataset#
fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels)= fashion_mnist.load_data()
#Creating the class names#
class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images=train_images/255
test_images=test_images/255
model=keras.Sequential([
     keras.layers.Flatten(input_shape=(28,28)),
     keras.layers.Dense(128, activation='relu'),
     keras.layers.Dense(10, activation='softmax')

])

#Compiling the model#
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
              
#Evaluating the model#
test_loss, test_acc= model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy:', test_acc)

#Predicting which class is the image#
predictions=model.predict(test_images)
print(class_names[np.argmax(predictions[1])])

#Plotting the prediction#
plt.figure()
plt.imshow(test_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
