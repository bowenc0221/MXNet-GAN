# MXNet-GAN
MXNet Implementation of various GAN, including: DCGAN [1], CGAN [2], Image-to-Image translation [3] (a.k.a. pix2pix)  

This is a working repo initially served as the final project for [UIUC ECE544NA](https://courses.engr.illinois.edu/ece544na/fa2017/_site/).  

## Prerequisites
- Linux (Tested in Ubuntu 16.04)
- Python 2 (You may need to modify some codes if you are using Python 3)
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Build MXNet from source (tested using MXNet version v.0.11.1).
  ```bash
  git clone --recursive https://github.com/apache/incubator-mxnet mxnet
  cd mxnet
  cp make/config.mk .
  vim config.mk  # You need to change configuration in order to enable cuda and cudnn 
  make -j8
  cd python
  sudo python setup.py install
  ```
- Clone this repo:
  ```bash
  git clone https://github.com/bowenc0221/MXNet-GAN
  cd MXNet-GAN
  ```
- Put MXNet python package into ```./external/mxnet/$(MXNET_VERSION)``` and modify ```MXNET_VERSION``` in ```./experiments/*.yaml``` to ```$(YOUR_MXNET_PACKAGE)```.
- Install python packages.
  ```bash
  pip install Cython
  pip install EasyDict
  pip install opencv-python
  ```

### DCGAN train/test
- Train
  ```bash
  python dcgan/train.py --cfg experiments/dcgan/mnist_dcgan.yaml
  ```
- Test
  ```bash
  python dcgan/test.py --cfg experiments/dcgan/mnist_dcgan.yaml
  ```
- Warning
  - I only implemented dcgan for mnist. You may need to write your own data iterator for other dataset.
  - I did not tune parameter for dcgan. I only trained for 1 epoch!

### CGAN train/test
- train
  ```bash
  python cgan/train.py --cfg experiments/cgan/mnist_cgan.yaml
  ```
- test
  ```bash
  python cgan/test.py --cfg experiments/cgan/mnist_cgan.yaml
  ```
- Warning
  - I only implemented dcgan for mnist. You may need to write your own data iterator for other dataset.
  - I did not tune parameter for dcgan. I only trained for 1 epoch!

### pix2pix train/test
- Download a pix2pix dataset (e.g.facades):
  ```bash
  bash ./datasets/download_pix2pix_dataset.sh facades
  ```
  Please refer to [pytorch-CycleGAN-and-pix2pix](https://github.com/bowenc0221/pytorch-CycleGAN-and-pix2pix) for dataset information.
- Train a model:
  - AtoB
    ```bash
    python pix2pix/train.py --cfg experiments/pix2pix/facades_pix2pix_AtoB.yaml
    ```
  - BtoA
    ```bash
    python pix2pix/train.py --cfg experiments/pix2pix/facades_pix2pix_BtoA.yaml
    ```
- Test a model:
  - AtoB
    ```bash
    python pix2pix/test.py --cfg experiments/pix2pix/facades_pix2pix_AtoB.yaml
    ```
  - BtoA
    ```bash
    python pix2pix/test.py --cfg experiments/pix2pix/facades_pix2pix_BtoA.yaml
    ```
- PatchGAN
  - You can use any PatchGAN listed in the paper by changing ```netD``` in configuration to ```'n_layers'``` and set ```n_layers``` to any number from 0-6.
  - ```n_layers = 0```: pixelGAN 1x1 discriminator
  - ```n_layers = 1```: patchGAN 16x16 discriminator
  - ```n_layers = 3```: patchGAN 70x70 discriminator (default setting in the paper)
  - ```n_layers = 6```: imageGAN 256x256 discriminator
- Train pix2pix on your own dataset
  - I only implemented pix2pix for cityscapes and facades dataset but you can generalize easily to your own dataset.
  - Prepare pix2pix-datasets according to [this link](https://github.com/bowenc0221/pytorch-CycleGAN-and-pix2pix/blob/master/README.md#pix2pix-datasets)
  - Modify ```num_train``` and ```num_val``` in ```./data/generate_train_val.py``` and run the script.
  - In configuration file, modify ```dataset``` part.
- Warning
  - Currently, I only implemented ```batch_size = 1```.

## Reference
[1] [DCGAN](https://arxiv.org/abs/1511.06434): Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks  
[2] [CGAN](https://arxiv.org/abs/1411.1784): Conditional Generative Adversarial Nets  
[3] [pix2pix](https://arxiv.org/abs/1611.07004): Image-to-Image Translation with Conditional Adversarial Networks  

## Acknowledgments
Code is inspired by:  
[1] [MXNet GAN Tutorial](https://mxnet.incubator.apache.org/tutorials/unsupervised_learning/gan.html)  
[2] [MXNet DCGAN Example](https://github.com/apache/incubator-mxnet/blob/master/example/gan/dcgan.py)  
[3] [A MXNet W-GAN Code](https://github.com/vsooda/mxnet-wgan)  
[4] [pytorch-CycleGAN-and-pix2pix](https://github.com/bowenc0221/pytorch-CycleGAN-and-pix2pix)  
