import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import os
import random
import pickle, datetime
from random import shuffle


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
from copy import deepcopy
from PIL import Image
import cv2


import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"]="0"
opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.015)
opts = tf.GPUOptions()
conf = tf.ConfigProto(gpu_options=opts)
conf.gpu_options.allow_growth = True

session = tf.Session(config=conf)

import numpy as np
KTF.set_session(session)



def featureTransform(resized):
    originalImg = resized
    YCC = deepcopy(resized)
    YCC = cv2.cvtColor(YCC,cv2.COLOR_BGR2LAB)[:,:,1:3]
    HSV = deepcopy(resized)
    HSV = cv2.cvtColor(HSV,cv2.COLOR_BGR2HSV)[:,:,0:3]
    Luv = deepcopy(resized)
    Luv = cv2.cvtColor(resized,cv2.COLOR_BGR2YUV)[:,:,0:1]
    tmp = np.concatenate((HSV,Luv,YCC),axis=2)
    return tmp


def extractLabel(Y):
    NUM_LABELS = 8
    NUM_DATA = len(Y)
    COLOR_DICT = {'blue':0,'green':1,'red':2,'yellow':3,'gray':4,'white':5,'black':6,'pink':7}
    out = []
    for l in Y:
        oneHot = np.array([0]*NUM_LABELS)
        oneHot[COLOR_DICT.get(l)] = 1
        out.append(oneHot)
    out = np.array(out).astype(np.float32)
    return out

def splitSet(X,Y):
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    val_X = []
    val_Y = []
    for idx,x in enumerate(X):
        seed = np.random.rand()
        if seed >0.4:
            train_X.append(x)
            train_Y.append(Y[idx])
        elif seed < 0.2:
            test_X.append(x)
            test_Y.append(Y[idx])
        else:
            val_X.append(x)
            val_Y.append(Y[idx])
            
    train_X = np.array(train_X).astype(np.float32)
    train_Y = extractLabel(train_Y)
    train_Y = train_Y.astype(dtype = np.float32)
    test_X = np.array(test_X).astype(np.float32)
    test_Y = extractLabel(test_Y)
    test_Y = test_Y.astype(dtype = np.float32)
    val_X = np.array(val_X).astype(np.float32)
    val_Y = extractLabel(val_Y)
    val_Y = val_Y.astype(dtype = np.float32)
    return train_X,train_Y,test_X,test_Y,val_X,val_Y

def getBatch(train_X,train_Y,batch_size):
    idx = range(0,len(train_Y))
    idxs = random.sample(idx, batch_size)
    return train_X[idxs],train_Y[idxs]


from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras import optimizers

CHANNELS = 6


model = Sequential()
model.add(Conv2D(64,(5,5),strides=(2,2),input_shape=(150,150,CHANNELS),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization(epsilon=0.001))
model.add(Dropout(0.7))

model.add(Conv2D(128,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization(epsilon=0.001))
model.add(Dropout(0.7))

model.add(Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization(epsilon=0.001))
model.add(Dropout(0.7))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(8,activation='softmax'))
model.summary()

allimg = glob.glob('./TestSet_01_perColor_500/*/*.jpg')
print len(allimg)
label = [i.split("/")[2] for i in allimg]

IMAGE_SIZE = 150
ori = []
X = [cv2.imread(imgName) for imgName in allimg]
inputX = []
for i in X:
    resized = cv2.resize(i,(IMAGE_SIZE,IMAGE_SIZE),interpolation = cv2.INTER_CUBIC)
    originalImg = resized
    tmp = featureTransform(resized)
    inputX.append(tmp)
    ori.append(originalImg)
Y = label


train_X,train_Y,test_X,test_Y,val_X,val_Y = splitSet(inputX,Y)

WEIGHT_DECAY = 0.0005
BATCH_SIZE = 3
LEARNING_RATE = 0.0001
DROPOUT = 0.5
ALPHA = 1e-4
BETA = 0.75
n = 5
k = 2
loss_metric = "categorical_crossentropy"



model.compile(loss=loss_metric, metrics=["accuracy"], 
              optimizer=optimizers.Nadam(lr=LEARNING_RATE,beta_1=0.9, beta_2=0.999, schedule_decay= WEIGHT_DECAY))




model.fit(train_X, train_Y,
           batch_size = BATCH_SIZE,
           epochs=300, 
           #verbose=2, callbacks=[histories])    
           validation_data = (test_X, test_Y), verbose=2)


model.save("./colorModel1019.h5")
