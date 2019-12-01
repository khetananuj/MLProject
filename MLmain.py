import numpy as np   
from PIL import Image
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt


filenames = os.listdir('./ST_200')

pic = []
X = []
y = []

for m in filenames:
    label = m[:-4].split('_')[1]
    if m != '.DS_Store':
        pic = Image.open('./ST_200/'+m)
        a = np.asarray(pic)
        a_array = pd.DataFrame(a)
        X.append(a)
        y.append(label)

Y = [[]for d in range (6800)]   

for n in range(6800):
    if y[n] == '1' or y[n] == '308':
        c = [1,0,0,0,0,0,0,0,0,0,0]
        Y[n]=c
    elif y[n] == '2' or y[n] == '203' or y[n] == '302' or y[n] == '422':
        c = [0,1,0,0,0,0,0,0,0,0,0]
        Y[n]=c
    elif y[n] == '3' or y[n] == '408' or y[n] == '417' or y[n] == '424':
        c = [0,0,1,0,0,0,0,0,0,0,0]
        Y[n]=c
    elif y[n] == '4':
        c = [0,0,0,1,0,0,0,0,0,0,0]
        Y[n]=c 
    elif y[n] == '5' or y[n] == '7' or y[n] == '402' or y[n] == '301':
        c = [0,0,0,0,1,0,0,0,0,0,0]
        Y[n]=c     
    elif y[n] == '6' or y[n] == '11' or y[n] == '12':
        c = [0,0,0,0,0,1,0,0,0,0,0]
        Y[n]=c  
    elif y[n] == '8' or y[n] == '204' or y[n] == '307' or y[n] == '401':
        c = [0,0,0,0,0,0,1,0,0,0,0]
        Y[n]=c  
    elif y[n] == '101':
        c = [0,0,0,0,0,0,0,1,0,0,0]
        Y[n]=c 
    elif y[n] == '102' or y[n] == '311' or y[n] == '404' or y[n] == '407':
        c = [0,0,0,0,0,0,0,0,1,0,0]
        Y[n]=c  
    elif y[n] == '303' or y[n] == '403' or y[n] == '416':
        c = [0,0,0,0,0,0,0,0,0,1,0]
        Y[n]=c  
    elif y[n] == '205' or y[n] == '306' or y[n] == '409' or y[n] == '410':
        c = [0,0,0,0,0,0,0,0,0,0,1]
        Y[n]=c
        
    

X = np.array(X,dtype = 'float32')    
X = X.reshape((6800,144,176,1))


Y = np.array(Y)
    

Y_train = []
X_train = []
for l in range (5440):
    trainy = Y[l]
    Y_train.append(trainy)
    trainx = X[l]
    X_train.append(trainx)
    
    
Y_test = []
X_test = []
    
for s in range (1360):
    testy = Y[s + 5440]
    Y_test.append(testy)
    testx = X[s + 5440]
    X_test.append(testx) 

Y_train = np.array(Y_train)
X_train = np.array(X_train)  
X_test = np.array(X_test)
Y_test = np.array(Y_test)

model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(144, 176,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(11, activation='softmax'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=6, batch_size=100, verbose=1, validation_data=(X_test, Y_test))

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
[loss, acc] = model.evaluate(X_test,Y_test, verbose = 1)
print("Accuracy:" + str(acc))

model.save('gesturerecognition.h5')
