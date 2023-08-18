# Re-basin via implicit Sinkhorn differentiation

Implementation of paper [Re-basin via implicit Sinkhorn differentiation](https://openaccess.thecvf.com/content/CVPR2023/papers/Pena_Re-Basin_via_Implicit_Sinkhorn_Differentiation_CVPR_2023_paper.pdf) (Accepted at CVPR 2023).

## Installation

    pip install sinkhorn-rebasin

## Running the examples 

|                          |                                                                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Basics                   | [![Basics](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a4NTjSUjIaai9oNtHtp1tZFvJjsGshpq?usp=sharing)                   |
| Models alignment         | [![Models alignment](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lDbYbbgyR4a9gJ8Lgoiz0DFB8OBouIDa?usp=sharing)         |
| Linear mode connectivity | [![Linear mode connectivity](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10tTRMuCGcmUGTKrnyFeyDyRXWGyy9PCu?usp=sharing) |

### Models alignment



    cd examples
    python main_alignment_{mlp|cnn|resnet}.py

| Example  | Layer from $\theta_A$                                                                                                         | Layer from $\pi_{\mathcal{P}}(\theta_A)$                                                                                           | Layer from $\theta_B$                                                                                                         |
| -------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| MLP      | ![Alignment modelA MLP](https://github.com/fagp/sinkhorn-rebasin/blob/main/resources/alignment_mlp_modelA.png?raw=true)       | ![Alignment pi_modelA MLP](https://github.com/fagp/sinkhorn-rebasin/blob/main/resources/alignment_mlp_pimodelA.gif?raw=true)       | ![Alignment modelB MLP](https://github.com/fagp/sinkhorn-rebasin/blob/main/resources/alignment_mlp_modelB.png?raw=true)       |
| VGG      | ![Alignment modelA VGG](https://github.com/fagp/sinkhorn-rebasin/blob/main/resources/alignment_cnn_modelA.png?raw=true)       | ![Alignment pi_modelA VGG](https://github.com/fagp/sinkhorn-rebasin/blob/main/resources/alignment_cnn_pimodelA.gif?raw=true)       | ![Alignment modelB VGG](https://github.com/fagp/sinkhorn-rebasin/blob/main/resources/alignment_cnn_modelB.png?raw=true)       |
| ResNet18 | ![Alignment modelA ResNet](https://github.com/fagp/sinkhorn-rebasin/blob/main/resources/alignment_resnet_modelA.png?raw=true) | ![Alignment pi_modelA ResNet](https://github.com/fagp/sinkhorn-rebasin/blob/main/resources/alignment_resnet_pimodelA.gif?raw=true) | ![Alignment modelB ResNet](https://github.com/fagp/sinkhorn-rebasin/blob/main/resources/alignment_resnet_modelB.png?raw=true) |


### Linear mode connectivity


    cd examples
    python main_lmc_{mlp|cnn|resnet}.py

| Dataset        | Model    | Accuracy LMC                                                | Cross Entropy Loss LMC                              |
| -------------- | -------- | ----------------------------------------------------------- | --------------------------------------------------- |
| Mnist          | MLP      | ![LMC MLP Accuracy](./resources/lmc_mlp_accuracy.gif)       | ![LMC MLP Loss](./resources/lmc_mlp_loss.gif)       |
| Mnist          | VGG      | ![LMC VGG Accuracy](./resources/lmc_cnn_accuracy.gif)       | ![LMC VGG Loss](./resources/lmc_cnn_loss.gif)       |
| Imagenette-320 | ResNet18 | ![LMC ResNet Accuracy](./resources/lmc_resnet_accuracy.gif) | ![LMC ResNet Loss](./resources/lmc_resnet_loss.gif) |