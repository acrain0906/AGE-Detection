
# =============================================================================
# Developer: Austin Crain
#      Date: 4 June 2020
#      Goal: Predict Age, Gender, and Ethnicity from images using deep neural 
#			 network.  
# =============================================================================

# import notify # homemade module for sending email/text notification at the 
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

from keras.applications import VGG16
from keras.layers import Dense
from keras.engine.training import Model
#from keras import backend as K
from tensorflow.keras import backend as K
import gc
import tensorflow as tf 
from tensorflow.compat.v1 import Session, RunOptions

import sys 
from twilio.rest import Client


# In[23]:


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




# Build Database
start = time.time()
# read images into list 
print('Summary 3...')
directory = './part3.tar/part3/'
summary3 = Summary(directory)

print('Summary 2...')
directory = './part3.tar/part2/'
summary2 = Summary(directory)

print('Summary 1...')
directory = './part3.tar/part1/'
summary1 = Summary(directory)
print('time taken: {} minutes'.format((time.time() - start)/60))

summary = summary1 # .append(summary2)


# In[25]:


detector = MTCNN()

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
	required_size=(224, 224)
	face_image = Image.fromarray(crop_img)
	face_image = face_image.resize(required_size)
	face_array = np.asarray(face_image)
	face_array = face_array/255

	return face_array


# In[26]:


print('cropping faces...')
start = time.time()
# create picture library
pictures = {}
for index, row in summary.iterrows():
	if index % 1000 == 0:
		print(index)
	pictures[row['filename']] = extractFace (catch(row['filename']))
print('time taken: {} minutes'.format((time.time() - start)/60))

del detector
K.clear_session()
gc.collect()


# In[29]:

# Get one hot encoding of columns B
def assignBin (x):
	binsize = 20
	return int(x / binsize)

summary['age'] = summary['age'].apply(assignBin)
maxBin = max(summary['age'].values) + 1
# one hot encoding 
def age2Bin(x): # 0 if row['gender']=='male' else 1
	vector = [0] * maxBin
	vector[x] = 1
	del vector[-1]
	return vector 

summary.ethnicity = pd.Categorical(summary.ethnicity)
summary.ethnicity = summary.ethnicity.cat.codes
maxBin2 = max(summary['ethnicity'].values) + 1
def eth2Bin(x):
	vector = [0] * maxBin2
	vector[x] = 1
	del vector[-1]
	return vector 



epochcnt = 128 # int(sys.argv[1] )
batch = 16 # int(sys.argv[2] )

# split into train and test data 
trainlen = split =  int(summary.shape[0] * 0.8)
summary = summary.sample(frac=1).reset_index(drop=True) # shuffle database rows in place
train = summary.iloc[:split]
test = summary.iloc[split:]

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
		gtrainy.append(0 if row['gender']=='male' else 1)
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

print('ratio: {}'.format(float(sum(gtrainy)) / len(gtrainy)))


print('build model...')

model = VGG16(
	include_top=True,
	weights="imagenet",
	input_shape=(224, 224, 3),
	classes=1000
)

del model.layers[-1] # delete top layer 
x = model.layers[-1].output
u = Dense(1024, activation='relu')(x)
age = Dense(maxBin - 1,activation='sigmoid', name = 'age')(u) # add one more layer
u = Dense(1024, activation='relu')(x)
gender = Dense(1,activation='sigmoid', name = 'gender')(u) # add one more layer
u = Dense(1024, activation='relu')(x)
ethnicity = Dense(maxBin2 - 1,activation='sigmoid', name = 'ethnicity')(u) # add one more layer
model = Model(model.input, [age, gender, ethnicity])
model.summary()

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"age": "binary_crossentropy",
	"gender": "binary_crossentropy",
	"ethnicity": "binary_crossentropy",
}


model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])

# train model 
print('train model...')
start = time.time()
model.fit(trainx, trainy, epochs=epochcnt, batch_size=batch) 
print('time taken: {} minutes'.format((time.time() - start)/60))

# serialize model to JSON
model_json = model.to_json()
with open("total model-{}-{}-{}.json".format(trainlen, epochcnt, batch), "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("total model-{}-{}-{}.h5".format(trainlen, epochcnt, batch))
print("Saved model to disk")


print('Collect testing data...')
trainx = []
atrainy = []
gtrainy = []
etrainy = []

for index, row in test.iterrows():
	img = pictures[row['filename']]
	if not isinstance(img, str):
		trainx.append(img)
		atrainy.append(age2Bin(row['age']))
		gtrainy.append(0 if row['gender']=='male' else 1)
		etrainy.append(eth2Bin(row['ethnicity']))
trainx = np.array(trainx)
atrainy = np.array(atrainy)
gtrainy = np.array(gtrainy).reshape(-1,1)
etrainy = np.array(etrainy)
print('ratio: {}'.format(float(sum(gtrainy)) / len(gtrainy)))

trainy = {
	"age": atrainy, 
	"gender": gtrainy, 
	"ethnicity": etrainy
}

# print('Test Accuracy: {}'.format())
scores = model.evaluate(trainx, trainy)
print('total time taken: {} minutes'.format((time.time() - startx)/60))
# sys.stdout = notify.log
for i in range(len(model.metrics_names)):
	print("%s: %.2f%%" % (model.metrics_names[i], scores[i]*100))
# notify.mail("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100), '')

