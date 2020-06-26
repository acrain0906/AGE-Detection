sudo yum install python3 python3-pip
sudo pip3 install --upgrade pip
python3 -m pip  ...
python3 -m pip install pandas numpy tensorflow-gpu keras opencv-python mtcnn keras_vggface twilio testfixtures 
python3 -m pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

python3 -m pip install cudnnenv

LD_LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LD_LIBRARY_PATH
CPATH=~/.cudnn/active/cuda/include:$CPATH
LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LIBRARY_PATH

cudnnenv install v7.6.4-cuda10
