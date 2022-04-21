# Relief Visualization GAN (RV-GAN)

This repository is for the GAN-based pretext for the paper titled "Self Supervised Learning for Semantic Segmentation of Archaeological 
Monuments in DTMs".

The code is mainly copied and adapted from [this](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) amazing repository. 

Create your own dataset module under `data/YOURDATASETNAME_dataset.py` by inheriting the `BaseDataset` class from 
`data/base_dataset.py` and customize it for your own dataset.

The code works with `Python 3.7.10` and `PyTorch 1.8.1`. 

Follow these steps:

- Clone this repository
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) and run the following
command to create the environment and install necessary libraries:

- `conda env create -f environment.yml`

This should create an environment called `ikg`. Activate the environment to train/test a model
on your own dataset:

- `conda activate ikg`

To train a model, edit the `options/base_options.py` and `options/train_options.py` file and run the following code for example:
- `python train_v2.py --name nameOfExperiment --model pix2pix --netG hrnet --netD basic --direction AtoB --lambda_L1 100 --dataset_mode YOURDATASETNAME --norm batch --pool_size 0 --gan_mode lsgan --gpu_ids 0,1,2,3 --batch_size 256`
  
This saves the model history and weights under `checkpoint/nameOfExperiment`


To test a model, run:
- `python evaluate_v2.py --name nameOfExperiment --model pix2pix --netG hrnet --direction AtoB --dataset_mode YOURDATASETNAME --crop_size 224 --epoch BEST_EPOCH --phase test`
  
`BEST_EPOCH` is a number indicating which epoch's saved weights for the generator to use for evaluation, e.g., `75`, in order to use 
epoch `75` weights from `checkpoint/nameOfExperiment/75_net_G.pth`

The pretrained model weights for the generator in RV-GAN are available [here](https://github.com/SSL-DTM/model_weights/releases/download/v0.0.0/RVGAN.pth). It can be used for initializing and finetuning semantic segmentation model. Check out the [semantic segmentation](https://github.com/SSL-DTM/semantic_segmentation) repo for how to do this.





