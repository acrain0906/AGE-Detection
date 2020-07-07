
from tensorflow.keras.applications import VGG16,VGG16, MobileNet, MobileNetV2, DenseNet121 # needs to be tensorflow.keras, otherwise wont accept data pipeline 
from tensorflow.keras.layers import Dense # needs to be 'keras' library, otherwise throws error
from tensorflow.keras import Model

def addOutputLayers(model, maxBin2, params):
	del model.layers[-1] # delete top layer 
	x = model.layers[-1].output
	u = Dense(params['age hidden layer'], activation='relu')(x)
	u = Dense(params['age hidden layer'], activation='relu')(u)
	u = Dense(params['age hidden layer 2'], activation='relu')(u)
	u = Dense(params['age hidden layer 2'], activation='relu')(u)
	u = Dense(params['age hidden layer 2'], activation='relu')(u)
	age = Dense(1,activation='linear', name = 'age')(u) # add one more layer
	u = Dense(params['gender hidden layer'], activation='relu')(x)
	u = Dense(params['gender hidden layer'], activation='relu')(u)
	u = Dense(int(params['gender hidden layer']/2), activation='relu')(u)
	gender = Dense(1,activation='sigmoid', name = 'gender')(u) # add one more layer
	u = Dense(params['ethnicity hidden layer'], activation='relu')(x)
	u = Dense(int(params['ethnicity hidden layer']/2), activation='relu')(u)
	ethnicity = Dense(maxBin2 - 1,activation='sigmoid', name = 'ethnicity')(u) # add one more layer
	model = Model(model.input, [age, gender, ethnicity])
	return model 

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
			"age": "mean_absolute_error",
			"gender": "binary_crossentropy",
			"ethnicity": "binary_crossentropy",
		}
		
metrics = {
			"age": "mean_absolute_error",
			"gender": "accuracy",
			"ethnicity": "accuracy",
		}

# =============================================================================
# VGG 
# =============================================================================
		
class vgg16:
	def __init__(self, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'VGG16-imagenet'
		
		self.model = VGG16(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'age hidden layer 2' : 128,
			'gender hidden layer' : 512,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin2, self.params)

		self.model.compile(optimizer='Adam', loss=losses, metrics=metrics)
	
	def name (self):
		return self._name
		
class vgg19:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'VGG19-imagenet'
		
		self.model = VGG19(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}
		
		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		
		# self.model.summary()
		
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name

# =============================================================================
# MOBILE 
# =============================================================================

class mobilenet:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'MobileNet-imagenet'
		
		self.model = MobileNet(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)

		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
			
class mobilenetv2:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'MobileNetV2-imagenet'
		
		self.model = MobileNetV2(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name

class nasnetlarge:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'NASNetLarge-imagenet'
		
		self.model = NASNetLarge(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
		
class nasnetmobile:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'NASNetMobile-imagenet'
		
		self.model = NASNetMobile(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
		
# =============================================================================
# RESNET 
# =============================================================================
	
class resnet50:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'ResNet50-imagenet'
		
		self.model = ResNet50(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
	
class resnet101:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'ResNet101-imagenet'
		
		self.model = ResNet101(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
	
class resnet152:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'ResNet152-imagenet'
		
		self.model = ResNet152(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
	
class resnet50v2:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'ResNet50V2-imagenet'
		
		self.model = ResNet50V2(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
	
class resnet101v2:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'ResNet101V2-imagenet'
		
		self.model = ResNet101V2(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
	
class resnet152v2:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'ResNet152V2-imagenet'
		
		self.model = ResNet152V2(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name

# =============================================================================
# DENSE 
# =============================================================================

class densenet121:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'DenseNet121-imagenet'
		
		self.model = DenseNet121(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name

class densenet169:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'DenseNet169-imagenet'
		
		self.model = DenseNet169(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name

class densenet201:
	def __init__(self, maxBin, maxBin2):
		"""
		must have self.model, self.params, and self._name
		"""
		self._name = 'DenseNet201-imagenet'
		
		self.model = DenseNet201(
			include_top=True,
			weights="imagenet",
			input_shape=(224, 224, 3),
			classes=1000
		)
		
		self.params = {
			'age hidden layer' : 512,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		self.model = addOutputLayers(self.model, maxBin, maxBin2, self.params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
		
VGG = [vgg16, vgg19]
MOBILE = [mobilenet, mobilenetv2, nasnetlarge, nasnetmobile]
RESNET = [resnet50, resnet101, resnet152, resnet50v2, resnet101v2, resnet152v2]
DENSE = [densenet121, densenet169, densenet201]

Total = VGG + MOBILE + RESNET + DENSE