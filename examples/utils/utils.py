from copy import deepcopy


def lerp(model1, model2, l):
    temporal_model = deepcopy(model1)
    for p, p1, p2 in zip(
        temporal_model.parameters(), model1.parameters(), model2.parameters()
    ):
        p.data.copy_((1 - l) * p1.data + l * p2.data)

    return temporal_model


def eval(model, dataset, criterion, device):
    model.to(device)
    cumulative_test_loss = 0
    total_test = 0
    model.eval()
    for x, y in dataset:
        z = model(x.to(device))
        loss_test = criterion(z, y.to(device))

        cumulative_test_loss += loss_test.item() * x.shape[0]
        total_test += x.shape[0]

    cumulative_test_loss /= total_test
    return cumulative_test_loss


def eval_loss_acc(model, dataset, criterion, device):
    model.to(device)
    cumulative_test_loss = 0
    cumulative_test_acc = 0
    total_test = 0
    model.eval()
    for x, y in dataset:
        z, loss = estep(x, y, model, criterion, device)
        acc_test = sum([1 if y[i] == z[i] else 0 for i in range(y.shape[0])])

        cumulative_test_loss += loss * x.shape[0]
        cumulative_test_acc += acc_test
        total_test += x.shape[0]

    cumulative_test_loss /= total_test
    cumulative_test_acc /= total_test
    return cumulative_test_loss, cumulative_test_acc


def estep(x, y, model, criterion, device):
    z = model(x.to(device))
    loss_test = criterion(z, y.to(device))
    return z.detach().argmax(1), loss_test.detach().item()
