#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np 
import os 
from PIL import Image 
import keras
import cv2
from keras import layers
from keras import models
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[ ]:





# In[8]:


lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('./leapGestrecog/leapGestRecog/00/'):
    lookup[j] = count
    reverselookup[count] = j
    count = count + 1
lookup


# In[9]:


x_data = []
y_data = []
datacount = 0 #  how many images are in our dataset
for i in range(0, 10): # ten top- folders
    for j in os.listdir('./leapGestrecog/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Aavoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('./leapGestrecog/leapGestRecog/0' + 
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('./leapGestrecog/leapGestRecog/0' + 
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
x_data = x_data.reshape((datacount, 120, 320, 1))
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) 
y_data = to_categorical(y_data)


# In[10]:


y_data.shape


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=None)


# In[12]:


y_train.shape


# In[14]:


from keras.layers import Dropout
model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.30))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[18]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)


# In[27]:


import matplotlib.pyplot as plt
print(model.evaluate(x_test,y_test))
print("Accuracy:" + str(acc))
model.summary()
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy on Training')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()
plt.plot(history.history['loss'])
plt.title('Model loss reduction on Training')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()


# In[ ]:


model.save('gesturerecognition.h5')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




