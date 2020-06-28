
from keras.applications import VGG16,VGG16, MobileNet, MobileNetV2, DenseNet121
from keras.layers import Dense
from keras.engine.training import Model

def addOutputLayers(model, maxBin, maxBin2, params):
	del model.layers[-1] # delete top layer 
	x = model.layers[-1].output
	u = Dense(params['age hidden layer'], activation='relu')(x)
	age = Dense(maxBin - 1,activation='sigmoid', name = 'age')(u) # add one more layer
	u = Dense(params['gender hidden layer'], activation='relu')(x)
	gender = Dense(1,activation='sigmoid', name = 'gender')(u) # add one more layer
	u = Dense(params['ethnicity hidden layer'], activation='relu')(x)
	ethnicity = Dense(maxBin2 - 1,activation='sigmoid', name = 'ethnicity')(u) # add one more layer
	model = Model(model.input, [age, gender, ethnicity])
	return model 

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
			"age": "binary_crossentropy",
			"gender": "binary_crossentropy",
			"ethnicity": "binary_crossentropy",
		}

# =============================================================================
# VGG 
# =============================================================================
		
class vgg16:
	def __init__(self, maxBin, maxBin2):
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)

		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}
		
		model = addOutputLayers(model, maxBin, maxBin2, params)
		
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)

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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
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
			'age hidden layer' : 1024,
			'gender hidden layer' : 1024,
			'ethnicity hidden layer' : 1024,
		}

		model = addOutputLayers(model, maxBin, maxBin2, params)
		self.model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
	
	def name (self):
		return self._name
		
VGG = [vgg16, vgg19]
MOBILE = [mobilenet, mobilenetv2, nasnetlarge, nasnetmobile]
RESNET = [resnet50, resnet101, resnet152, resnet50v2, resnet101v2, resnet152v2]
DENSE = [densenet121, densenet169, densenet201]

Total = VGG.extend(MOBILE).extend(RESNET).extend(DENSE)