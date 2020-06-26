# AGE-Detection
Determine Age, Gender, and Ethnicity from images of faces using a deep neural network.  Images basted on UTKFace dataset.  

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

### Setup
Initial Setup on AWS AMI 2 System: 
```
sudo yum install python3 python3-pip
sudo pip3 install --upgrade pip
python3 -m pip  ...
python3 -m pip install pandas numpy tensorflow-gpu keras opencv-python mtcnn keras_vggface twilio testfixtures 
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

## Overview
### Extracting Faces
Extracted Faces using pre-built MTCNN which is based on the [Multi-task Cascade Convolutional Network](https://arxiv.org/abs/1604.02878)

### Normalizing Images
For better results from the deep neural network, the most likely image from each picture was reshaped into a 224x224 image, with each pixel ranging from zero to one.  

### Designing Model 
To speed up training time, transfer learning was employed.  The model is based on 'imagenet' 

## Results 
Test Acuracy
  * Age: 99.8%
  * Gender: 80.5%
  * Ethnicity: 91.62%



