# Differentiable re-basin

Implementation of paper [Re-basin via implicit Sinkhorn differentiation](https://arxiv.org/abs/2212.12042) (Accepted at CVPR 2023).

## Installation

``pip install -e .``

## Running the examples

### Models aligment



    cd examples
    python main_alignment_{mlp|cnn|resnet}.py

| Example  | Layer from $\theta_A$                                               | Layer from $\pi_{\mathcal{P}}(\theta_A)$                                 | Layer from $\theta_B$                                               |
| -------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| MLP      | ![Alignment modelA MLP](./resources/alignment_mlp_modelA.png)       | ![Alignment pi_modelA MLP](./resources/alignment_mlp_pimodelA.png)       | ![Alignment modelB MLP](./resources/alignment_mlp_modelB.png)       |
| VGG      | ![Alignment modelA VGG](./resources/alignment_cnn_modelA.png)       | ![Alignment pi_modelA VGG](./resources/alignment_cnn_pimodelA.png)       | ![Alignment modelB VGG](./resources/alignment_cnn_modelB.png)       |
| ResNet18 | ![Alignment modelA ResNet](./resources/alignment_resnet_modelA.png) | ![Alignment pi_modelA ResNet](./resources/alignment_resnet_pimodelA.png) | ![Alignment modelB ResNet](./resources/alignment_resnet_modelB.png) |

### Linear mode connectivity



    cd examples
    python main_lmc_{mlp|cnn|resnet}.py

| Dataset        | Model    | Accuracy LMC                                                | Cross Entropy Loss LMC                              |
| -------------- | -------- | ----------------------------------------------------------- | --------------------------------------------------- |
| Mnist          | MLP      | ![LMC MLP Accuracy](./resources/lmc_mlp_accuracy.png)       | ![LMC MLP Loss](./resources/lmc_mlp_loss.png)       |
| Mnist          | VGG      | ![LMC VGG Accuracy](./resources/lmc_cnn_accuracy.png)       | ![LMC VGG Loss](./resources/lmc_cnn_loss.png)       |
| Imagenette-320 | ResNet18 | ![LMC ResNet Accuracy](./resources/lmc_resnet_accuracy.png) | ![LMC ResNet Loss](./resources/lmc_resnet_loss.png) |