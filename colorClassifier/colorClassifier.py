import keras
from keras.models import Sequential,load_model
import os
import random
import cv2
from copy import deepcopy
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
import glob


os.environ["CUDA_VISIBLE_DEVICE"] = "0"
opts=tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
conf=tf.ConfigProto(gpu_options=opts)
conf.gpu_options.allow_growth = True




class colorClassifier():

	def __init__(self,modelPath):
		os.environ["CUDA_VISIBLE_DEVICE"] = "0"
		opts=tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
		conf=tf.ConfigProto(gpu_options=opts)
		conf.gpu_options.allow_growth = True
		session = tf.Session(config=conf)
		KTF.set_session(session)
		self.model = load_model(modelPath)
		self.color_axis = ['blue','green','red','yellow','gray','white','black','pink']
		
	
	def featureTransform(self,resized):
    		YCC = deepcopy(resized)
   		YCC = cv2.cvtColor(YCC,cv2.COLOR_BGR2LAB)[:,:,1:3]
    		HSV = deepcopy(resized)
    		HSV = cv2.cvtColor(HSV,cv2.COLOR_BGR2HSV)[:,:,0:3]
    		Luv = deepcopy(resized)
    		Luv = cv2.cvtColor(resized,cv2.COLOR_BGR2YUV)[:,:,0:1]
    		tmp = np.concatenate((HSV,Luv,YCC),axis=2)
    		return tmp

	def getColor(self,img):
		resized = cv2.resize(img,(150,150),interpolation=cv2.INTER_CUBIC)
		tmp = self.featureTransform(resized)
		x = np.expand_dims(tmp,axis=0)
		predictions = np.argmax(self.model.predict(x),1)
                return self.color_axis[predictions[0]]  

	def getColorWithConfidence(self,img,confidence):
		resized = cv2.resize(img,(150,150),interpolation=cv2.INTER_CUBIC)
		tmp = self.featureTransform(resized)
		x = np.expand_dims(tmp,axis=0)
		predictions = self.model.predict(x)
		if np.amax(predictions)>confidence:
		    return self.color_axis[np.argmax(predictions,1)[0]]
		else:
		    return "none"
	def getColorConfidence(self,img):
		res = {}
		resized = cv2.resize(img,(150,150),interpolation=cv2.INTER_CUBIC)
		tmp = self.featureTransform(resized)
		x = np.expand_dims(tmp,axis=0)
		predictions = self.model.predict(x)
                for i in range(len(self.color_axis)):
			res[self.color_axis[i]] = predictions[0][i]
		
		return res


if __name__ == "__main__":
	modelPath = '/home/superuser/pva-faster-rcnn/tools/colorModel1025.h5'
	classifier = colorClassifier(modelPath)
	confidence = 0.9
	allimg = glob.glob('./TestSet_01_perColor_500/*/*.jpg')
	print len(allimg)
	label = [i.split("/")[2] for i in allimg]
	inp = cv2.imread(allimg[np.random.randint(0,len(allimg))])
	res = classifier.getColor(inp)
	resConf = classifier.getColorConfidence(inp)
        resWithConf = classifier.getColorWithConfidence(inp,confidence)
	print "groundTruth = {} and result = {}".format(label[3],res)
 	print "confidence = {}".format(resConf)
	print "res with confidence {} is {}".format(confidence,resWithConf)
