# pytorch-planet-amazon
PyTorch models and training code for 'Planet: Understanding the Amazon from Space' Kaggle

## Overview
This is my model training, validation, and inference code for the above mentioned Kaggle competition. Most components should be reusable for other image classification applications. 

Image augmentation via the Dataset class is done using OpenCV and custom cropping methods as opposed to the usual Torchvision PIL based methods. This was to allow more flexibility in augmentation, including arbitrary rotations and downscaling beyond image bounds without introducing black or fixed border artifacts. 

Aside from a submission file handling bug that impacted private leaderboard scores of our teams submissions, these models, when ensembled together with some model, resolution, and train/validate fold variation, are capable of results in the top 2% of contestants.

## Models 
Models used where pulled in from a variety of sources and can be created with a factory method *model_factory.create_model()*. 

For all models I used I added dropout as it helped reduce overfit with a limited size dataset. I also changed the pooling layer to a dynamic option that could change with different resolutions and allow me to switch between global average, max, and a combination of max and average pooling. 

### DensNet and ResNet, https://github.com/pytorch/vision/
I started with the DenseNet and ResNet models and pretrained weights available in the Torchvision module. I ended up duplicating the code here to fix one issue with kwargs passthrough for DenseNet and added dynamic avg/max pooling and dropout (ResNet).

### Inception-V4 and Inception-Resnet-V2, https://github.com/Cadene/tensorflow-model-zoo.torch 
I used the Inception models and pretrained weights in Cadene's model zoo. A fix was applied for a batch norm momentum of 0.

### Facebook ResNet-200 Torch, https://github.com/facebook/fb.resnet.torch
From Facebook's Torch implementation, I imported pretrained weights and model using https://github.com/clcarwin/convert_torch_to_pytorch 

### Wide-Resnet (WRN-50-2), https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/
From szagoruyko's implementation, I imported pretrained weights and model using convert_torch_to_pytorch. I did not end up using this as the license is unspecified.

### Facebook ResNext-101-32x4d, https://github.com/facebookresearch/ResNeXt 
From Facebook's implementation, I imported pretrained weights and model using convert_torch_to_pytorch but ended up scraping the weights due to CC BY-NC 4.0 license.

## Usage Examples

Train a ResNet-50 model from pretrained weights with 256x256 image size, on CV fold 1 with fairly typical SGD params and 30% dropout:
 
    python train.py /data/amazon/train-jpg --batch-size 32 --opt sgd --lr 0.01 --model resnet50 --loss mlsm --fold 1 
        --img-size 256 --drop 0.3 --pretrained

Train a DenseNet-201 model from pretrained weights with 256x256 image size, on CV fold 1 with ADAM optimizer and 10% dropout, start by fine-tuning only fully connected layer for half an epoch with SGD:
 
    python train.py /data/amazon/train-jpg --batch-size 32 --opt adam --lr 0.0001 --model densenet201 --loss mlsm 
        --ft-epochs 0.5 --ft-opt sgd --ft-lr 0.0005 --fold 1 --img-size 256 --drop 0.1 --pretrained

Run inference on a trained Inception-V4 model with test data, 256x256 resolution, using 8x test-time augmentation:

    python inference.py /data/amazon/test-jpg --batch-size 128 --model inception_v4 --img-size 299 
        --restore-checkpoint output/train/checkpoint-21.pth.tar --tta 8

## Dense-Sparse-Dense

I tried implementing the ideas in https://arxiv.org/abs/1607.04381 

Generally, going through a sparse-dense cycle seemed to increase overfitting with the competition dataset. Possibly due to a lack in variation and datasets size when compared to a dataset like Imagenet. Not much time was spent validating my approach.

To try:

1. Train model normally for some time until loss plateaus. Stop training.

2. Resume training and sparsify with the following command (you may also want to tweak LR or push out decay epochs):

            python train.py /data/amazon/train-jpg --batch-size 32 --opt sgd --lr 0.01 --model resnet101 --loss mlsm 
                --drop 0.1 --resume checkpout-21.pth.tar --sparse

3. Stop after loss recovers

4. Resume training and densify with the following (same as above but, specify a sparse checkpoint and take away the added sparse argument):

            python train.py /data/amazon/train-jpg --batch-size 32 --opt sgd --lr 0.01 --model resnet101 --loss mlsm 
                --drop 0.1 --resume checkpout-29.pth.tar

5. Stop when loss plateaus. I found test loss had already bottomed and started climbing by this point.



   
