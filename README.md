# Model Compression in Deep Vision Networks

## Project Description

 - Current state-of-the-art vision models have millions of parameters which makes them very resource-intensive. Our motivation is to reduce the computational and memory requirements of these models.
 - Our goal is to compress and optimize these Convolutional Neural Network (CNN) models by reducing their size, inference time, and memory consumption, without compromising much on accuracy.
   This would improve the model accessibility and it can be deployed to embedded and edge devices which have limited resources and compute constraints
 - Pruning and Quantization can make the model smaller, faster, simpler, and more efficient, and lead to improved performance. Pruning can make the model simpler by removing unnecessary connections and weights, and Quantization can make the model more efficient by reducing the number of computations.
 - We use pruning and quantization approaches to compress the model. As a result, we observe that model size, inference time, and memory consumption metrics are reduced significantly.
   We compute layer importance in order to perform layer-importance based pruning, and then compare performance at different pruning levels. We also experiment with different quantization techniques and compare metrics with each approach.


## Repo Description

> We worked with ResNet18 and VGG16 Architectures on the CIFAR-10 dataset
```
   .
   ├── VGG16                   # Folder containing all VGG16 and VGG16_BN code and notebooks
   │   ├── Torch_Pruning       # Contains modules for filter pruning in Conv layers
   │   │   └── ...             # Code Referenced from https://github.com/VainF/Torch-Pruning with modifications             
   │   ├── layer_importance    # Contains modules for computing importance of Conv layers
   │   │   └── ...             # Code Referenced from https://github.com/tyui592/Pruning_filters_for_efficient_convnets with modifications              
   │   ├── *.ipynb             # Contains all notebooks for layer importance computation, pruning, fixed-point quantization, and static quantization for VGG16 and VGG16_BN
   ├── ResNet18                # Folder containing all ResNet18 code and notebooks
   │   ├── Torch_Pruning       # Contains modules for filter pruning in Conv layers
   │   │   └── ...             # Code Referenced from https://github.com/VainF/Torch-Pruning with modifications             
   │   ├── quantize_new        # Contains code for training and fixed point quantization of ResNet18 model
   │   │   └── ...             # Code Referenced from https://github.com/aaron-xichen/pytorch-playground with modifications             
   │   ├── *.ipynb             # Contains all notebooks for pruning, fixed-point quantization, static quantization, quantization aware training for ResNet18
   └── assets
       └── image*.png          # Contains all results and table images for README

```

## Commands to Execute

 - All the Jupyter Notebooks are directly executable and contain the required imports to necessary modules.

## Results 

### ResNet 18
- ![Alt text](assets/image1.png?raw=true "Title")


### VGG16

### VGG16_BN

## Observations

 - Text Here

## References:
 - https://github.com/VainF/Torch-Pruning
 - https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
 - https://github.com/Forggtensky/Quantize_Pytorch_Vgg16AndMobileNet
 - https://github.com/aaron-xichen/pytorch-playground
 - https://github.com/leimao/PyTorch-Static-Quantization
 - https://github.com/tyui592/Pruning_filters_for_efficient_convnets


