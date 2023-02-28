import torch
from models import MLP2
from rebasin.loss import MidLoss
from rebasin import RebasinNet
from copy import deepcopy
from utils import train, lerp, eval_loss_acc
from datasets.classification import MNistDataset, SmallMNistDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# this code is similar to experiment 2 of our paper
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu"):
    print("Consider using GPU, if available, for a significant speed up.")

# preparing dataset
dataset = SmallMNistDataset(root="minist_data", download=True, train=True)
training, validation = torch.utils.data.random_split(
    dataset,
    [9000, 1000],
    generator=torch.Generator().manual_seed(1),
)
test = MNistDataset(root="minist_data", download=True, train=False)
dataset_train = torch.utils.data.DataLoader(
    training, batch_size=1000, shuffle=True, num_workers=0
)
dataset_val = torch.utils.data.DataLoader(
    validation, batch_size=1000, shuffle=False, num_workers=0
)
dataset_test = torch.utils.data.DataLoader(
    test, batch_size=1000, shuffle=False, num_workers=0
)

# model A - trained on Mnist
modelA = MLP2(num_hidden=2, num_inputs=28 * 28, num_classes=10, dropout_p=0.0)
print("Training network A")
modelA = train(
    modelA,
    dataset_train,
    dataset_val,
    torch.optim.AdamW(modelA.parameters(), lr=0.001),
    torch.nn.CrossEntropyLoss(),
    device,
    50,
)
loss, acc = eval_loss_acc(modelA, dataset_test, torch.nn.CrossEntropyLoss(), device)
print("Model A: test loss {:1.3f}, test accuracy {:1.3f}".format(loss, acc))

# model B - trained on Mnist
modelB = MLP2(num_hidden=2, num_inputs=28 * 28, num_classes=10, dropout_p=0.25)
print("\nTraining network B")
modelB = train(
    modelB,
    dataset_train,
    dataset_val,
    torch.optim.AdamW(modelB.parameters(), lr=0.001),
    torch.nn.CrossEntropyLoss(),
    device,
    50,
)
loss, acc = eval_loss_acc(modelB, dataset_test, torch.nn.CrossEntropyLoss(), device)
print("Model B: test loss {:1.3f}, test accuracy {:1.3f}".format(loss, acc))


# rebasin network for model A
pi_modelA = RebasinNet(modelA, input=torch.zeros(1, 28 * 28).to(device))
pi_modelA.to(device)

# mid point loss
criterion = MidLoss(modelB, criterion=torch.nn.CrossEntropyLoss())

# optimizer for rebasin network
optimizer = torch.optim.AdamW(pi_modelA.p.parameters(), lr=0.1)

print("\nTraining Re-Basing network")
for iteration in range(20):
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
    if cumulative_val_loss == 0:
        break

pi_modelA.eval()
rebased_model = deepcopy(pi_modelA())

lambdas = torch.linspace(0, 1, 50)
costs_naive = torch.zeros_like(lambdas)
costs_lmc = torch.zeros_like(lambdas)
acc_naive = torch.zeros_like(lambdas)
acc_lmc = torch.zeros_like(lambdas)

print("\nComputing interpolation for LMC visualization")
for i in tqdm(range(lambdas.shape[0])):
    l = lambdas[i]

    temporal_model = lerp(rebased_model, modelB, l)
    costs_lmc[i], acc_lmc[i] = eval_loss_acc(
        temporal_model, dataset_test, torch.nn.CrossEntropyLoss(), device
    )

    temporal_model = lerp(modelA, modelB, l)
    costs_naive[i], acc_naive[i] = eval_loss_acc(
        temporal_model, dataset_test, torch.nn.CrossEntropyLoss(), device
    )

plt.figure()
plt.plot(lambdas, costs_naive, label="Naive")
plt.plot(lambdas, costs_lmc, label="Sinkhorn Re-basin")
plt.title("Loss")
plt.xticks([0, 1], ["ModelA", "ModelB"])
plt.legend()
# plt.show()
plt.savefig("lmc_mlp_loss.png")

plt.figure()
plt.plot(lambdas, acc_naive, label="Naive")
plt.plot(lambdas, acc_lmc, label="Sinkhorn Re-basin")
plt.title("Accuracy")
plt.xticks([0, 1], ["ModelA", "ModelB"])
plt.legend()
# plt.show()
plt.savefig("lmc_mlp_accuracy.png")

print("LMC for MLP!")
