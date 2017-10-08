# MXNet-GAN
MXNet Implementation of various GAN, including: DCGAN [1], CGAN [2], Image-to-Image translation [3] (a.k.a. pix2pix)  

This repo initially serves as the final project for [UIUC ECE544NA](https://courses.engr.illinois.edu/ece544na/fa2017/_site/).  

## Prerequisites
- Linux (Tested in Ubuntu 16.04)
- Python 2 (You may need to modify some codes if you are using Python 3)
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Build MXNet from source.
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
- Make a directory named ```external/mxnet/$MXNET_VERSION``` and put ```$MXNET/python/mxnet``` in this directory.

### DCGAN train/test

### CGAN train/test

### pix2pix train/test
- Download a pix2pix dataset (e.g.facades):
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
Please refer to [pytorch-CycleGAN-and-pix2pix](https://github.com/bowenc0221/pytorch-CycleGAN-and-pix2pix)
- Train a model:
```bash
python pix2pix/train.py --cfg experiments/mxnet_pix2pix.yaml
```
- Test a model:
```bash
python pix2pix/test.py --cfg experiments/mxnet_pix2pix.yaml
```

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
