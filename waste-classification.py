# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:00:55 2019

@author: ACER
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from PIL import Image

#Image.open(r"C:\Users\ACER\Desktop\AI\DATASET\TEST\O\O_13191.jpg")

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

train_gen = ImageDataGenerator(rescale = 1./255)
test_gen = ImageDataGenerator(rescale = 1./255)

training_set = train_gen.flow_from_directory(r"C:\Users\ACER\Desktop\AI\DATASET\TRAIN",target_size = (64,64),batch_size = 32,class_mode = 'binary')
test_set = test_gen.flow_from_directory(r"C:\Users\ACER\Desktop\AI\DATASET\TEST",target_size = (64,64),batch_size = 32,class_mode = 'binary')

model.fit_generator(training_set,steps_per_epoch = 706,epochs = 10,validation_data = test_set,validation_steps = 79)

#im = Image.open(r"C:\Users\ACER\Desktop\AI\DATASET\TEST\R\R_10798.jpg")
#im.show()
test_image = image.load_img(r"C:\Users\ACER\Desktop\AI\DATASET\TEST\O\O_13192.jpg", target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = model.predict(test_image)
print(training_set.class_indices)
if (result[0][0]==1):
    print("Recyclable")
else:
    print("Organic")




