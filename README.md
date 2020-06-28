# AGE-Detection
Determine Age, Gender, and Ethnicity from images of faces using a deep neural network.  Images basted on UTKFace dataset.  This report covers various models, pre- and post-processing.  

## Running the Model
### Required Libraries 
  *  pandas 
  *  numpy 
  *  tensorflow-gpu 
  *  keras 
  *  opencv-python # for extracting faces from images
  *  mtcnn # for detecting faces and locations im
  *  keras_vggface 
  *  twilio # for texting 
  *  testfixtures 
  * imblearn

### Setup
Initial Setup on AWS AMI 2 System: 
```
sudo yum install python3 python3-pip
sudo pip3 install --upgrade pip
python3 -m pip  ...
python3 -m pip install pandas numpy tensorflow-gpu keras opencv-python mtcnn keras_vggface twilio testfixtures imblearn
python3 -m pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U
python3 -m pip install cudnnenv
```

On each startup, run this code:
``` batch
LD_LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LD_LIBRARY_PATH
CPATH=~/.cudnn/active/cuda/include:$CPATH
LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LIBRARY_PATH
cudnnenv install v7.6.4-cuda10
```

To run, launch the test code to train the model and generate a result summary:
`> python3 AGE-Detect.py`

## Overview
### Extracting Faces
Extracted Faces using pre-built MTCNN which is based on the [Multi-task Cascade Convolutional Network](https://arxiv.org/abs/1604.02878)

### Normalizing Images
For better results from the deep neural network, the most likely image from each picture was reshaped into a 224x224 image, with each pixel ranging from zero to one. 

Also, even out the ratio of the outputs (i.e. men/women) as to prevent the model from prefering to predict any class in particular.  This rebalancing of inputs was accomplished using under-sampling, since oversampling would increase the memory requirements beyond the current limit.  

### Designing Model 
To speed up training time, transfer learning was employed.  Each of the following models is weighted using 'imagenet' as a baseline.  Afterwards, several dense layers leading to one of three outputs are appended.  These outputs determine the Age, Gender, and Ethnicity of the person in the image.  The models compared were:
  * VGG16
  * VGG19
  * MobileNet
  * MobileNetv2
  * NASNetlarge
  * NASNetMobile
  * resnet50
  * resnet101
  * resnet152
  * resnet50v2
  * resnet101v2
  * resnet152v2
  * densenet121
  * densenet169
  * densenet201

## Results 
Test Accuracy for VGG16
  * Age: 99.8%
  * Gender: 80.5%
  * Ethnicity: 91.62%



