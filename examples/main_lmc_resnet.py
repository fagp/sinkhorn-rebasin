import torch

from rebasin.loss import RndLoss
from rebasin import RebasinNet
from copy import deepcopy
from datasets.classification import SubsetImageNetDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import train, lerp, eval_loss_acc
from time import time
import torchvision.transforms as tr
import os
import torchvision

LOAD_TRAINED_MODEL = False
if not os.path.exists("./data/imagenette2-320"):
    print(
        "Please download the imagenet subset first using the script download_imagenet_subset.sh"
    )
    os.system("bash download_imagenet_subset.sh")
    # exit(-1)

# this code is similar to experiment 2 of our paper
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu"):
    print("Consider using GPU, if available, for a significant speed up.")

# preparing dataset
num_classes = 10
dataset = SubsetImageNetDataset(
    root="data/imagenette2-320/train",
    transform=tr.Compose(
        [
            tr.ToTensor(),
            tr.CenterCrop(320),
            tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            tr.RandomHorizontalFlip(),
            tr.RandomRotation(5),
            tr.RandomVerticalFlip(),
            tr.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            tr.Resize(224),
        ]
    ),
)
training, validation = torch.utils.data.random_split(
    dataset,
    [8469, 1000],
    generator=torch.Generator().manual_seed(1),
)
test = SubsetImageNetDataset(
    root="data/imagenette2-320/val",
    transform=tr.Compose(
        [
            tr.ToTensor(),
            tr.CenterCrop(320),
            tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            tr.Resize(224),
        ]
    ),
)

dataset_train = torch.utils.data.DataLoader(
    training, batch_size=128, shuffle=True, num_workers=2
)
dataset_val = torch.utils.data.DataLoader(
    validation, batch_size=100, shuffle=False, num_workers=2
)
dataset_test = torch.utils.data.DataLoader(
    test, batch_size=128, shuffle=False, num_workers=2
)

# model A - trained on subset of imagenet
modelA = torchvision.models.resnet18(pretrained=True)
modelA.fc = torch.nn.Linear(512, num_classes)

if LOAD_TRAINED_MODEL:
    sd = torch.load("model1.pt")
    modelA.load_state_dict(sd["weights"])
else:
    print("Training network A")
    modelA = train(
        modelA,
        dataset_train,
        dataset_val,
        torch.optim.AdamW(modelA.parameters(), lr=0.001),
        torch.nn.CrossEntropyLoss(),
        device,
        20,
    )
for p in modelA.modules():
    if isinstance(p, torch.nn.BatchNorm2d):
        p.track_running_stats = False
        p.running_mean = None
        p.running_var = None
loss, acc = eval_loss_acc(modelA, dataset_test, torch.nn.CrossEntropyLoss(), device)
print("Model A: test loss {:1.3f}, test accuracy {:1.3f}".format(loss, acc))

# model B - trained on subset of imagenet
modelB = torchvision.models.resnet18(pretrained=False)
modelB.fc = torch.nn.Linear(512, num_classes)
if LOAD_TRAINED_MODEL:
    sd = torch.load("model2.pt")
    modelB.load_state_dict(sd["weights"])
else:
    print("\nTraining network B")
    modelB = train(
        modelB,
        dataset_train,
        dataset_val,
        torch.optim.AdamW(modelB.parameters(), lr=0.001),
        torch.nn.CrossEntropyLoss(),
        device,
        30,
    )
for p in modelB.modules():
    if isinstance(p, torch.nn.BatchNorm2d):
        p.track_running_stats = False
        p.running_mean = None
        p.running_var = None
loss, acc = eval_loss_acc(modelB, dataset_test, torch.nn.CrossEntropyLoss(), device)
print("Model B: test loss {:1.3f}, test accuracy {:1.3f}".format(loss, acc))

# rebasin network for model A
pi_modelA = RebasinNet(
    modelA, input_shape=(1, 3, 224, 224), permutation_type="broadcast"
)
pi_modelA.to(device)

# rand point loss
criterion = RndLoss(modelB, criterion=torch.nn.CrossEntropyLoss())

# optimizer for rebasin network
optimizer = torch.optim.AdamW(pi_modelA.p.parameters(), lr=0.1)

print("\nTraining Re-Basing network")
t1 = time()
for iteration in range(50):
    # training step
    pi_modelA.train()
    cumulative_train_loss = 0
    total_train = 0
    for x, y in dataset_train:
        rebased_model = pi_modelA()
        loss_training = criterion(rebased_model, x.to(device), y.to(device))

        optimizer.zero_grad()
        loss_training.backward()
        optimizer.step()  # only updates the permutation matrices

        cumulative_train_loss += loss_training.item() * x.shape[0]
        total_train += x.shape[0]

    cumulative_train_loss /= total_train

    cumulative_val_loss = 0
    total_val = 0
    # validation step
    pi_modelA.eval()
    for x, y in dataset_train:
        rebased_model = pi_modelA()
        loss_validation = criterion(rebased_model, x.to(device), y.to(device))

        cumulative_val_loss += loss_validation.item() * x.shape[0]
        total_val += x.shape[0]

    cumulative_val_loss /= total_val

    print(
        "Iteration {:02d}: loss training {:1.3f}, loss validation {:1.3f}".format(
            iteration, cumulative_train_loss, cumulative_val_loss
        )
    )
    if cumulative_train_loss == 0:
        break

print("Elapsed time {:1.3f} secs".format(time() - t1))

pi_modelA.update_batchnorm(modelA)
pi_modelA.eval()
rebased_model = deepcopy(pi_modelA())
rebased_model.eval()

lambdas = torch.linspace(0, 1, 20)
costs_naive = torch.zeros_like(lambdas)
costs_lmc = torch.zeros_like(lambdas)
acc_naive = torch.zeros_like(lambdas)
acc_lmc = torch.zeros_like(lambdas)

temporal_model = deepcopy(modelA)

print("\nComputing interpolation for LMC visualization")
for i in tqdm(range(lambdas.shape[0])):
    l = lambdas[i]

    temporal_model = lerp(rebased_model, modelB, l, temporal_model)
    costs_lmc[i], acc_lmc[i] = eval_loss_acc(
        temporal_model, dataset_train, torch.nn.CrossEntropyLoss(), device
    )

    temporal_model2 = lerp(modelA, modelB, l, temporal_model)
    costs_naive[i], acc_naive[i] = eval_loss_acc(
        temporal_model2, dataset_train, torch.nn.CrossEntropyLoss(), device
    )

plt.figure()
plt.plot(lambdas, costs_naive, label="Naive")
plt.plot(lambdas, costs_lmc, label="Sinkhorn Re-basin")
plt.title("Loss")
plt.xticks([0, 1], ["ModelA", "ModelB"])
plt.legend()
# plt.show()
plt.savefig("lmc_resnet_loss.pdf")

plt.figure()
plt.plot(lambdas, acc_naive, label="Naive")
plt.plot(lambdas, acc_lmc, label="Sinkhorn Re-basin")
plt.title("Accuracy")
plt.xticks([0, 1], ["ModelA", "ModelB"])
plt.legend()
# plt.show()
plt.savefig("lmc_resnet_accuracy.pdf")

print("LMC for ResNet!")
