
# =============================================================================
# Developer: Austin Crain
#      Date: 4 June 2020
#      Goal: Predict Age, Gender, and Ethnicity from images using deep neural 
#			 network.  
# =============================================================================
# %autoreload 2

import notify # homemade module for sending email/text notification at the 
				# end of the script with results/timing.  
				
				# https://arxiv.org/abs/1604.02878

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
# from keras.utils.vis_utils import plot_model
import json 
from PIL import Image
import numpy as np
import time
startx = time.time()

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import random

import tensorflow as tf 
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet152V2
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
#from keras import backend as K
from tensorflow.keras import backend as K
import gc
from tensorflow.compat.v1 import Session, RunOptions

import sys 
from twilio.rest import Client

from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from model import VGG, RESNET, DENSE, MOBILE, Total
from report import generateReport


from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler


import statistics
import os
 
# from tensorflow.keras.preprocessing.image import image_dataset_from_directory



def Summary(directory):
	data = []
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
			'age' : int(params[0].split('\\')[-1]),
			'gender' : gender,
			'ethnicity' : ethnicity,
			'filename' : file
		})
		
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
	print(summary3.shape)

	print('Summary 2...')
	directory = base_directory + 'part2/'
	summary2 = Summary(directory)
	print(summary2.shape)

	print('Summary 1...')
	# directory = 'D:/Downloads/Faces/AWS Instance/data/part1/' windows
	directory = base_directory + 'part1/'
	summary1 = Summary(directory)
	print(summary1.shape)
	print('time taken: {} minutes'.format((time.time() - start)/60))

	summary = summary3.append(summary2).append(summary1)
	print(summary.shape)
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



def saveModel(model):
	# serialize model to JSON
	model_json = model.to_json()
	with open("total model-{}-{}-{}.json".format(trainlen, epochcnt, batch), "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("total model-{}-{}-{}.h5".format(trainlen, epochcnt, batch))
	print("Saved model to disk")


		
def loadFaces(summary):
	print('loading faces...')
	start = time.time()
	# create picture library
	pictures = {}
	for index, row in summary.iterrows():
		if index % 1000 == 0:
			print(index)
		pictures[row['filename']] = np.asarray(Image.fromarray(cv2.imread(row['filename'])))
	timing = (time.time() - start)/summary.shape[0] *1000
	print('time taken: {} minutes'.format((time.time() - start)/60))
	return pictures, timing
	
	
def code (a, g, e):
    # combine categories into digits, ones for ethnicity, tens for gender, and 100s for age (digits layed out: aaage)
	return a * 100 + g * 10 + e 
	
def decode(age):
	e = age % 10 # the bottom digit represents the ethnicity 0-4
    
    # the next digit represents the gender 0-1
	age //= 10 
	g = age % 10
    
    # the rest represents the age
	age //= 10 
	return age, g, e 


def getXandY(train):
    """ 
    This function extracts the inputs and outputs from a pandas dataframe and a dictionary containing all images.  
  
    For each row in the dataframe, the age, gender, and ethnicity are extracted to form an output vector.  
    The each row's filename is used to load an image as the input vector.  
  
    Parameters: 
    train (pandas.Dataframe): Dataframe containing the columns 'age', 'gender', 'ethnicity', and 'filename'
  
    Returns: 
    numpy.ndarray: input vector for model 
             dict: output vectors for model 
    
    """
    
    trainx = [] # initialize input array 
    
    # initialize 3 output arrays
    age = []
    gender = []
    ethnicity = []
    
    
    for index, row in train.iterrows(): # go through dataset rows
        img = pictures[row['filename']] # collect image
        if not isinstance(img, str): # some images are a string 'No Face', remove these
            trainx.append(img.flatten()) # append the image as a row 
            
            # append each of the output columns (age, gender, ethnicity)
            age.append(row['age']) 
            gender.append(row['gender'])
            ethnicity.append(row['ethnicity'])
    
    # convert arrays to Numpy ndarray
    trainx = np.array(trainx)
    age = np.array(age)
    gender = np.array(gender)
    ethnicity = np.array(ethnicity)
    
    # convert outputs into one dictionary with labels
    trainy = {
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity
    }
    
    return trainx, trainy
    
# collect sample, groupby labels, count, and calculate the variance.  This tells you how the ratio of the labels vary.  
def variance (encodey):
    sample = encodey.tolist()
    counts = [sample.count(x) for x in set(sample)]
    print("Variance of sample set is % s" %(statistics.variance(counts))) 

    print(encodey.shape)
    # print(decodex.shape)
    
def saveFaces(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    for key, value in pictures.items():
        # results += '  - {} \t:  {}\n'.format(key,value)
        try:
            # print(type(value))
            cv2.imwrite(directory + key.split('\\')[-1], value*255)
            # cv2.imwrite('color_img.jpg', img)
            #print(value.shape)
            #cv2.imshow("image", value)
            #cv2.waitKey()
            #break;
        except:
            print(key)
            print(type(value))
            print(value)
            

def convert2Softmax(data):
    binsize = max(data) + 1
    odata = []
    for x in data:
        vector = [0] * binsize
        vector[x] = 1
        del vector[-1]
        odata.append(vector)
    return np.array(odata) 
            
if __name__ == '__main__':
	# import pandas and dict
	saved = True
	if saved:
		directory = '.\\testdata\\'  # 'D:\\Downloads\\Faces\\data\\'
		summary = Summary(directory)  # ('./data/')
		pictures, load_time = loadFaces(summary)
	else:
		summary = importData  ('D:\\Downloads\\Faces\\part3.tar\\')# ('./part3.tar/')
		# k = importData ('.\\test\\') # ('./test/')

		detector = MTCNN()
		pictures, crop_time = cropFaces(summary)
		# kpictures, _ = cropFaces(k)
		del detector

		# save pictures 
		directory = 'D:\\Downloads\\Faces\\data\\'
		saveFaces(directory)
		
	# convert to numpy array 

	# convert gender from string to int
	summary['gender'] = summary['gender'].apply(lambda x: 1 if x == 'male' else 0)

	# convert ethnicity from string to int
	summary['temp'] = summary['ethnicity'].astype('category') # convert to categorical variable, but preserve string
	summary['ethnicity'] = summary['temp'].cat.codes # convert categories to int
	# k['ethnicity'] = k['ethnicity'].astype('category').cat.codes # do both steps together in test data 
	num_ethnicities = max(summary['ethnicity'].values) + 1 # keep track of number of ethnicities for softmax
	ethnicity = summary[['ethnicity', 'temp']].drop_duplicates().sort_values('ethnicity') # get lookup of ethnicity id to string

	targetNames = {
		'age' : None,
		'gender' : ['male', 'female'],
		'ethnicity' : list(ethnicity['temp'].values)
	}

	# split into train and test data 
	trainlen = split =  int(summary.shape[0] * 0.8)
	summary = summary.sample(frac=1).reset_index(drop=True) # shuffle database rows in place
	train = summary.iloc[:split]
	test = summary.iloc[split:]

	# convert to numpy (not resampled)
	testx, testy = getXandY(test)
	testy['age'] = np.array(testy['age'])
	testy['gender'] = convert2Softmax(testy['gender'])
	testy['ethnicity'] = convert2Softmax(testy['ethnicity'])
	# kx, ky = generateSoftMax2(k) 

	# convert to numpy array (X) and dict of numpy arrays (Y)
	trainx, trainy = getXandY(train)

	# encode y for resample 
	encodey = np.array([
		code(
			trainy['age'][i], 
			trainy['gender'][i], 
			trainy['ethnicity'][i]
		) for i in range (trainy['age'].shape[0])
	])
	encodey.shape

	# resample
	variance(encodey)
	rus = RandomUnderSampler(random_state=42)
	#ros = RandomOverSampler(random_state=42)
	trainx, encodey = rus.fit_resample(trainx, encodey)
	variance(encodey)

	# decode
	decodey = {
		'age' : [], 
		'gender' : [], 
		'ethnicity' : []
	}

	decodex = []
	for row in trainx:
		decodex.append(row.reshape((224, 224, 3)))
	decodex = np.array(decodex)
		
	for row in encodey:
		a, g, e = decode(row.tolist())
		decodey['age'].append(a)
		decodey['gender'].append(g)
		decodey['ethnicity'].append(e)
		
	decodey['age'] = np.array(decodey['age'])
	decodey['gender'] = convert2Softmax(decodey['gender'])
	decodey['ethnicity'] = convert2Softmax(decodey['ethnicity'])

	
	epochcnt = 25 # 40 # int(sys.argv[1] )
	batch = 16 # int(sys.argv[2] )
	
	# convert to pre-fetch

	def generator():
		for s1, s2, s3, l in zip(decodey['age'].flatten().tolist(), decodey['gender'], decodey['ethnicity'], decodex):
			yield l, {"age": s1, "gender": s2, 'ethnicity' : s3}

	ds_train = tf.data.Dataset.from_generator(generator, output_types=(tf.int64, {"age": tf.int64, "gender": tf.int64, "ethnicity": tf.int64}))

	ds_train = ds_train.cache()
	ds_train = ds_train.shuffle(min(decodex.shape[0], 1000)) 
	ds_train = ds_train.batch(batch)
	ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

	# Generate Test tensorflow dataset 

	def generator2():
		for s1, s2, s3, l in zip(testy['age'], testy['gender'], testy['ethnicity'], testx):
			yield l, {"age": s1, "gender": s2, 'ethnicity' : s3}

	ds_test = tf.data.Dataset.from_generator(generator2, output_types=(tf.int64, {"age": tf.int64, "gender": tf.int64, "ethnicity": tf.int64}))

	ds_test = ds_test.batch(batch)
	ds_test = ds_test.cache()
	ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
	
	print('build model...')
	for model_class in  [VGG[0]]:
		#try:
		start2 = time.time()
		age = model_class(num_ethnicities)
		print('model: {}'.format(age.name()))

		# train model 
		print('train model...')
		start = time.time()
		training_history = age.model.fit(ds_train, epochs=epochcnt, validation_data=ds_test)     
		print(type(training_history))
		training_time = (time.time() - start)/60
		print('training time taken: {} minutes'.format(training_time))


		results, acc = generateReport(age.model, testx, testy, targetNames, training_history, training_time, crop_time, age.params)
		notify.mail('{} Finished: {}'.format(acc*100, age.name()), results)
		with open('{}-results.txt'.format(age.name()), 'w') as f:
			f.write(results)


		# Test pictures 
		result = age.model.evaluate(kx, ky)
		r = ''
		for i, value in enumerate(result):
			r += '  - {} \t:  {}\n'.format(age.model.metrics_names[i],value)
		notify.mail('results', r)

		#except Exception as e:
		#	print('Error!')
		#	print(e)

	print('total time taken: {} minutes'.format((time.time() - startx)/60))

