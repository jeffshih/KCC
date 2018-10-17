import keras
from keras.models import Sequential
import os
import random
import cv2
from copy import deepcopy
from keras.model import load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np



os.environ["CUDA_VISIBLE_DEVICE"] = "0"
opts=tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
conf=tf.ConfigProto(gpu_options=opts)
conf.gpu_options.allow_growth = True




class colorClassifier(modelPath):

	def __init__(self,modelPath):
		os.environ["CUDA_VISIBLE_DEVICE"] = "0"
		opts=tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
		conf=tf.ConfigProto(gpu_options=opts)
		conf.gpu_options.allow_growth = True
		session = tf.Sesstion(config=conf)
		KTF.set_session(session)
		self.model = load_model(modelPath)
		self.color_axis = ['blue','green','red','yellow','gray','white','black','pink']
		
	
	def featureTransform(self,resized):
    		originalImg = resized
    		YCC = deepcopy(resized)
   		YCC = cv2.cvtColor(YCC,cv2.COLOR_BGR2LAB)[:,:,1:3]
    		HSV = deepcopy(resized)
    		HSV = cv2.cvtColor(HSV,cv2.COLOR_BGR2HSV)[:,:,0:3]
    		Luv = deepcopy(resized)
    		Luv = cv2.cvtColor(resized,cv2.COLOR_BGR2YUV)[:,:,0:1]
    		tmp = np.concatenate((HSV,Luv,YCC),axis=2)
    		return tmp

	def getColor(self,img):
		resized = cv2.resized(img,(150,150),interpolation=cv2.INTER_CUBIC)
		tmp = featureTransform(resized)
		x = np.expand_dims(tmp,axis=0)
		predictions = model.predict(x)
		return self.color_axis[predictions]  



if __name__ == "__main__":
	model = './colorModel1017.h5'
	classifier = colorClassifier(model)
	allimg = glob.glob('./TestSet_01_perColor_500/*/*.jpg')
	print len(allimg)
	label = [i.split("/")[2] for i in allimg]
	inp = cv2.imread(allimg[3])
	res = classifier.getColor(inp)
	print "label = {} and result = {}".format(label[3],res)
