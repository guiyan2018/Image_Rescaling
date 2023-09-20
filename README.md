# 1.paper:Enhancing Image Rescaling Using High Frequency Guidance and Attentions in Downscaling and Upscaling Network

# 2.Dependencies and Installation
# 2.1. create conda virtual env.
    Python 3 (Recommend to use Anaconda)
# 2.2. install pytorch, reference url: https://pytorch.org.
    PyTorch >= 1.0
    NVIDIA GPU + CUDA

# 2.3. install dependent packages.
    pip install numpy opencv-python lmdb pyyaml

# 2.4. install tensorBoard
    PyTorch >= 1.1: pip install tb-nightly future
    PyTorch == 1.0: pip install tensorboardX

# 3.Dataset Preparation
Commonly used training and testing datasets can be downloaded here. 
    
    url:  https://pan.baidu.com/s/1HlM1Mex-Glgd76ZnB42bFA?pwd=vq97  提取码：vq97

# 4. Training and testing codes are in 'codes/'.

# 4.1. Training
First set a config file in options/train/, then run as following:

	python train.py -opt options/train/train_DSNetSRNet_x2.yml

# 4.2. Test
First set a config file in options/test/, then run as following:

	python test.py -opt options/test/test_DSNetSRNet_x2.yml

#### Code Framework
The code framework follows [BasicSR](https://github.com/xinntao/BasicSR/tree/master/codes). It mainly consists of four parts - `Config`, `Data`, `Model` and `Network`.

Let us take the train command `python train.py -opt options/train/train_DSNetSRNet_x2.yml` for example. A sequence of actions will be done after this command. 

- [`train.py`](./train.py) is called. 
- Reads the configuration in [`options/train/train_DSNetSRNet_x2.yml`](/home/ps/workspace/xieyan/DSNet_SRNet/codes/options/train/train_DSNetSRNet_x2.yml), including the configurations for data loader, network, loss, training strategies and etc. The config file is processed by [`options/options.py`](./options/options.py).
- Creates the train and validation data loader. The data loader is constructed in [`data/__init__.py`](./data/__init__.py) according to different data modes.
- Creates the model (is constructed in [`models/__init__.py`](./models/__init__.py). 
- Start to train the model. Other actions like logging, saving intermediate models, validation, updating learning rate and etc are also done during the training.  

#### Config
#### [`options/`](./options) Configure the options for data loader, network structure, model, training strategies and etc.

#### Data
#### [`data/`](./data) A data loader to provide data for training, validation and testing.

#### Model
#### [`models/`](./models) Construct models for training and testing.

#### Network
#### [`models/modules/`](./models/modules) Construct network architectures.

