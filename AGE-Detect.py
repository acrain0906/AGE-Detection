
# =============================================================================
# Developer: Austin Crain
#      Date: 4 June 2020
#      Goal: Predict Age, Gender, and Ethnicity from images using deep neural 
#			 network.  
# =============================================================================

import notify # homemade module for sending email/text notification at the 
				# end of the script with results/timing.  

from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

from mtcnn import MTCNN
import cv2
import glob
import pandas as pd
from keras.utils.vis_utils import plot_model
import json 
from PIL import Image
import numpy as np
import time
startx = time.time()

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import random

from keras.applications import VGG16, InceptionV3, ResNet152V2
from keras.layers import Dense
from keras.engine.training import Model
#from keras import backend as K
from tensorflow.keras import backend as K
import gc
import tensorflow as tf 
from tensorflow.compat.v1 import Session, RunOptions

import sys 
from twilio.rest import Client

from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from model import VGG, RESNET, DENSE, MOBILE, Total
from report import generateReport
 
# from tensorflow.keras.preprocessing.image import image_dataset_from_directory



def Summary(directory):
	data = []
	i = 0
	for file in glob.glob("{}*.jpg".format(directory)):
		# from the file name, determine properties:
		params = file.replace(directory,'').split('_')

		# determine gender
		gender = 'male'
		if params[1] == '1':
			gender = 'female'
		
		# determine ethnicity
		ethnicity = 'white'
		if params[2] == '1':
			ethnicity = 'black'
		if params[2] == '2':
			ethnicity = 'asian'
		if params[2] == '3':
			ethnicity = 'indian'
		if params[2] == '4':
			ethnicity = 'other'
		
		# generate dataframe row 
		data.append({
			'age' : int(params[0]),
			'gender' : gender,
			'ethnicity' : ethnicity,
			'filename' : file
		})
		
		i+= 1 
		if i > 1000: 
			break;
	return pd.DataFrame(data)

# In[24]:

def importData(base_directory):
	# Build Database
	start = time.time()
	# read images into list 
	print('Summary 3...')
	directory = base_directory + 'part3/'
	# print(directory)
	summary3 = Summary(directory)

	print('Summary 2...')
	directory = base_directory + 'part2/'
	# summary2 = Summary(directory)

	print('Summary 1...')
	# directory = 'D:/Downloads/Faces/AWS Instance/data/part1/' windows
	directory = base_directory + 'part1/'
	# summary1 = Summary(directory)
	print('time taken: {} minutes'.format((time.time() - start)/60))

	summary = summary3 # .append(summary2)
	return summary 

# pixels = summary['picture'].values[200][0]
def catch (file):
	return [cv2.imread(file)] # read in an image as a numpy arry

def checkBounds(wall, x):
	if x > wall:
		x = wall
	if x < 0:
		x = 0
	return x

def extractFace (pixels):
	pixels = pixels[0]
	Height = pixels.shape[0]
	Width = pixels.shape[1]
	img = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
	data = detector.detect_faces(img) # detect faces

	if len(data) == 0 or data is None:
		print('no faces found')
		return '[No Faces]'

	box = data[0]
	conf = 0.0
	for box in data:
		x, y, width, height = box['box']
		x2, y2 = x + width, y + height
		if box['confidence'] > conf:
			x = checkBounds(Width, x)
			x2 = checkBounds(Width, x2)
			y = checkBounds(Height, y)
			y2 = checkBounds(Height, y2)
			crop_img = img[y:y2, x:x2] # extract faces
			conf = box['confidence']
		
		rectangle(pixels, (x, y), (x2, y2), (255,0,255), 3)
	
	try:
		crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
	except:
		print(type(crop_img))
		print(crop_img.shape)
		print('({}, {}) ({}, {})'.format(x, y, x2, y2))
		imshow('face detection', pixels)
		waitKey(0)
		destroyAllWindows()
	# resize pixels to the model size
	required_size= (224, 224)
	face_image = Image.fromarray(crop_img)
	face_image = face_image.resize(required_size)
	face_array = np.asarray(face_image)
	face_array = face_array/255

	return face_array

def cropFaces(summary):
	print('cropping faces...')
	start = time.time()
	# create picture library
	pictures = {}
	for index, row in summary.iterrows():
		if index % 1000 == 0:
			print(index)
		pictures[row['filename']] = extractFace (catch(row['filename']))
	timing = (time.time() - start)/summary.shape[0] *1000
	print('time taken: {} minutes'.format((time.time() - start)/60))
	return pictures, timing

# In[29]:

# Get one hot encoding of columns B
def assignBin (x):
	return int(x / binsize)



# one hot encoding 
def age2Bin(x): 
	vector = [0] * maxBin
	vector[x] = 1
	del vector[-1]
	 # TODO?
	return vector

def gender2Bin(x):
	vector = [0, 0]
	vector[0 if x == 'male' else 1] = 1
	del vector[-1]
	return vector

def eth2Bin(x):
	vector = [0] * maxBin2
	vector[x] = 1
	del vector[-1]
	return vector 

def generateSoftMax(train):
	# convert training data 
	print('Collect training data...')
	trainx = []
	atrainy = []
	gtrainy = []
	etrainy = []

	for index, row in train.iterrows():
		img = pictures[row['filename']]
		if not isinstance(img, str):
			trainx.append(img)
			atrainy.append(age2Bin(row['age']))
			gtrainy.append(gender2Bin(row['gender']))
			etrainy.append(eth2Bin(row['ethnicity']))
	trainx = np.array(trainx)
	atrainy = np.array(atrainy)
	gtrainy = np.array(gtrainy).reshape(-1,1)
	etrainy = np.array(etrainy)
	trainy = {
		"age": atrainy, 
		"gender": gtrainy, 
		"ethnicity": etrainy
	}
	return trainx, trainy

# calculate results 
# TODO_01


def saveModel(model):
	# serialize model to JSON
	model_json = model.to_json()
	with open("total model-{}-{}-{}.json".format(trainlen, epochcnt, batch), "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("total model-{}-{}-{}.h5".format(trainlen, epochcnt, batch))
	print("Saved model to disk")
	

	
# print('Test Accuracy: {}'.format())
# scores = model.evaluate(trainx, trainy)
# confusion_matrix(y_test, y_pred)

# sys.stdout = notify.log
#string = '{}\n'.format(time.time() - start2)
#for i in range(len(model.metrics_names)):
#	string += "%s: %.2f%%\n" % (model.metrics_names[i], scores[i]*100)
	
if __name__ == '__main__':
	summary = importData('./part3.tar/')
	
	detector = MTCNN()
	pictures, crop_time = cropFaces(summary)
	del detector
	
	
	#K.clear_session()
	#gc.collect()
	binsize = 5
	max_age = float(max(summary['age'].values))
	summary['age'] = summary['age'].apply(assignBin)
	maxBin = max(summary['age'].values) + 1
	age_names = []
	for i in range (maxBin):
		age_names.append('{}-{}'.format(i * binsize, (i+1) * binsize))
	
	
	summary['temp'] = summary['ethnicity'].astype('category') #pd.Categorical(summary.ethnicity)
	summary['ethnicity'] = summary['temp'].cat.codes
	maxBin2 = max(summary['ethnicity'].values) + 1
	
	df = summary[['ethnicity', 'temp']].drop_duplicates().sort_values('ethnicity')
	print(type(df))
	# TODO_01
	targetNames = {
		'age' : age_names,
		'gender' : ['male', 'female'],
		'ethnicity' : list(df['temp'].values)
	}
	
	print(targetNames)

	epochcnt = 1 # 40 # int(sys.argv[1] )
	batch = 16 # int(sys.argv[2] )

	# split into train and test data 
	trainlen = split =  int(summary.shape[0] * 0.8)
	summary = summary.sample(frac=1).reset_index(drop=True) # shuffle database rows in place
	train = summary.iloc[:split]
	test = summary.iloc[split:]
	trainx, trainy = generateSoftMax(train)
	print('ratio: {}'.format(float(sum(trainy['gender'])) / len(trainy['gender'])))
	
	print('build model...')
	for model_instance in Total:
		start2 = time.time()
		age = AGE(maxBin, maxBin2)

		# train model 
		print('train model...')
		start = time.time()
		training_history = age.model.fit(trainx, trainy, epochs=epochcnt, batch_size=batch) 
		print(type(training_history))
		training_time = (time.time() - start)/60
		print('training time taken: {} minutes'.format(training_time))
		
		testx, testy = generateSoftMax(test)
		results, acc = generateReport(age.model, testx, testy, targetNames, training_history, training_time, crop_time, age.params)
		notify.mail('{} Finished: {}'.format(acc*100, age.name()), results)
		with open('{}-results.txt'.format(age.name()), 'w') as f:
			f.write(results)
	
	print('total time taken: {} minutes'.format((time.time() - startx)/60))
	
	




	