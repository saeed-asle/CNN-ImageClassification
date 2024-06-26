import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
import keras
#Data link : https://www.kaggle.com/puneet6060/intel-image-classification

trainpath = 'yourpath/seg_train/'
testpath = 'yourpath/seg_test/'
predpath = 'yourpath/seg_pred/'

for folder in  os.listdir(trainpath + 'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train/' + folder + '/*.jpg'))
    print(f'For training data , found {len(files)} in folder {folder}')

for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(pathname= str( testpath +'seg_test/' + folder + '/*.jpg'))
    print(f'For testing data , found {len(files)} in folder {folder}')

code = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x 

size = []
for folder in  os.listdir(trainpath +'seg_train') : 
    files = gb.glob(os.path.join(trainpath, 'seg_train', folder, '*.jpg'))
    num_images_to_select = int(len(files) * 0.2)
    selected_files = random.sample(files, num_images_to_select)
    for file in selected_files:  
        image = plt.imread(file)
        size.append(image.shape)
series = pd.Series(size, dtype='float64')
print(series.value_counts())

size = []
for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(os.path.join(testpath, 'seg_test', folder, '*.jpg'))
    num_images_to_select = int(len(files) * 0.09)
    selected_files = random.sample(files, num_images_to_select)
    for file in selected_files: 
        image = plt.imread(file)
        size.append(image.shape)
series = pd.Series(size, dtype='float64')
print(series.value_counts())

size = []
for folder in  os.listdir(predpath +'seg_pred') : 
    files = gb.glob(pathname= str( predpath +'seg_pred/' + folder + '/*.jpg'))
    num_images_to_select = int(len(files) * 0.09)
    selected_files = random.sample(files, num_images_to_select)
    for file in selected_files: 
        image = plt.imread(file)
        size.append(image.shape)
series = pd.Series(size, dtype='float64')
print(series.value_counts())

s=100

X_train = []
y_train = []
for folder in  os.listdir(trainpath +'seg_train') : 
    files = gb.glob(os.path.join(trainpath, 'seg_train', folder, '*.jpg'))
    num_images_to_select = int(len(files) * 0.19)
    selected_files = random.sample(files, num_images_to_select)
    for file in selected_files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(code[folder])

print(f'we have {len(X_train)} items in X_train')

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    plt.title(getcode(y_train[i]))

X_test = []
y_test = []
for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(os.path.join(testpath, 'seg_test', folder, '*.jpg'))
    num_images_to_select = int(len(files) * 0.09)
    selected_files = random.sample(files, num_images_to_select)
    for file in selected_files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_test.append(list(image_array))
        y_test.append(code[folder])
print(f'we have {len(X_test)} items in X_test')
   
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_test[i])    
    plt.axis('off')
    plt.title(getcode(y_test[i]))

X_pred = []
files = gb.glob(pathname= str(predpath + 'seg_pred/*.jpg'))
num_images_to_select = int(len(files) * 0.05)
selected_files = random.sample(files, num_images_to_select)
for file in selected_files: 
    image = cv2.imread(file)
    image_array = cv2.resize(image , (s,s))
    X_pred.append(list(image_array))       
print(f'we have {len(X_pred)} items in X_pred')

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')

X_train = np.array(X_train)
X_test = np.array(X_test)
X_pred_array = np.array(X_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)


print(f'X_train shape  is {X_train.shape}')
print(f'X_test shape  is {X_test.shape}')
print(f'X_pred shape  is {X_pred_array.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')

KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(s,s,3)),
        keras.layers.Conv2D(150,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Conv2D(120,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(80,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(50,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Flatten() ,    
        keras.layers.Dense(120,activation='relu') ,    
        keras.layers.Dense(100,activation='relu') ,    
        keras.layers.Dense(50,activation='relu') ,        
        keras.layers.Dropout(rate=0.5) ,            
        keras.layers.Dense(6,activation='softmax') ,    
        ])
KerasModel.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print('Model Details are : ')
print(KerasModel.summary())

# Check if there are saved model weights
if os.path.exists('model_weights.h5'):
    print("Model weights file 'model_weights.h5' already exists. Skipping training.")
else:
    # Train the model
    print("Training the model...")
    epochs = 20  # Set your desired number of epochs
    ThisModel = KerasModel.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=1)
    # Save the trained model weights
    KerasModel.save_weights('model_weights.h5')
epochs = 20

# Check if the model has already been trained
if 'ThisModel' not in locals():
    ThisModel = KerasModel.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=1)

ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)

print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy))

y_pred = KerasModel.predict(X_test)

print('Prediction Shape is {}'.format(y_pred.shape))

y_result = KerasModel.predict(X_pred_array)

print('Prediction Shape is {}'.format(y_result.shape))

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')
    plt.title(getcode(np.argmax(y_result[i])))
