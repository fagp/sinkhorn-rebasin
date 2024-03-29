import torch
from rebasin.loss import DistL1Loss
from rebasin import RebasinNet, matching
from utils import visualize_kernels
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
import torchvision

# this code is similar to experiment 1 of our paper
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu"):
    print("Consider using GPU, if available, for a significant speed up.")

# model A pretrained on ImageNet
modelA = torchvision.models.resnet18(pretrained=True, num_classes=1000)
modelA.to(device)

# rebasin network for model A
pi_modelA = RebasinNet(modelA, input_shape=(1, 3, 224, 224))
pi_modelA.to(device)


# we will create a random permuation of A
# this will be model B
pi_modelA.random_init()
pi_modelA.eval()
modelB = deepcopy(pi_modelA())
modelB.to(device)

# storing the permutation matrix P0 for comparison purposes
target = pi_modelA.p[0].data.clone().cpu().numpy().astype("uint8")

# we set the permutation matrices to be the identity again
del pi_modelA
pi_modelA = RebasinNet(modelA, input_shape=(1, 3, 224, 224))
pi_modelA.to(device)
pi_modelA.identity_init()
pi_modelA.train()
print("\nMaking sure we initialize the permutation matrices to I")
print(pi_modelA.p[0].data.clone().cpu().numpy().astype("uint8"))
print("\n")
# distance loss
criterion = DistL1Loss(modelB)

# optimizer for rebasin network
optimizer = torch.optim.AdamW(pi_modelA.p.parameters(), lr=10.0)

# try to find the permutation matrices that originated modelB
print("\nTraining Re-Basing network")
t1 = time()
for iteration in range(50):
    # training step
    pi_modelA.train()  # this uses soft permutation matrices
    rebased_model = pi_modelA()
    loss_training = criterion(rebased_model)  # this compared rebased_model with modelB

    optimizer.zero_grad()
    loss_training.backward()
    optimizer.step()  # only updates the permutation matrices

    # validation step
    pi_modelA.eval()  # this uses hard permutation matrices
    rebased_model = pi_modelA()
    loss_validation = criterion(rebased_model)
    print(
        "Iteration {:02d}: loss training {:1.3f}, loss validation {:1.3f}".format(
            iteration, loss_training, loss_validation
        )
    )
    if loss_validation == 0:
        break

print("Elapsed time {:1.3f} secs".format(time() - t1))

# if loss validation is 0, then we found the same permutation matrix
pi_modelA.eval()
estimated = matching(pi_modelA.p[0].data.clone().cpu()).numpy().astype("uint8")

print("\nTarget permutation matrix P0:")
print(target)
target = visualize_kernels(torch.from_numpy(target).float())
plt.imshow(target)
plt.axis("off")
# plt.show()
plt.savefig("alignment_resnet_permutation_target.png")

print("\nEstimated permutation matrix P0:")
print(estimated)
estimated = visualize_kernels(torch.from_numpy(estimated).float())
plt.imshow(estimated)
plt.axis("off")
# plt.show()
plt.savefig("alignment_resnet_permutation_estimated.png")

if loss_validation == 0:
    print("\nTransportation plan found for ResNet!")

# visualize the kernels of the first convolutional layer
kernelsA = visualize_kernels(modelA.conv1.weight)
plt.imshow(kernelsA)
plt.axis("off")
# plt.show()
plt.savefig("alignment_resnet_modelA.png")

kernelsB = visualize_kernels(modelB.conv1.weight)
plt.imshow(kernelsB)
plt.axis("off")
# plt.show()
plt.savefig("alignment_resnet_modelB.png")

kernelspiA = visualize_kernels(pi_modelA().conv1.weight)
plt.imshow(kernelspiA)
plt.axis("off")
# plt.show()
plt.savefig("alignment_resnet_pimodelA.png")
