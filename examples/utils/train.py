from tqdm import tqdm


def train(model, dataset_train, dataset_val, optimizer, criterion, device, epochs=5):
    model.to(device)

    for epoch in tqdm(range(epochs)):
        cumulative_train_loss = 0
        total_train = 0
        model.train()
        for x, y in dataset_train:
            tloss = tstep(x, y, model, optimizer, criterion, device)
            cumulative_train_loss += tloss * x.shape[0]
            total_train += x.shape[0]

        cumulative_train_loss /= total_train

        cumulative_val_loss = 0
        total_val = 0
        model.eval()
        for x, y in dataset_val:
            vloss = vstep(x, y, model, criterion, device)
            cumulative_val_loss += vloss * x.shape[0]
            total_val += x.shape[0]

        cumulative_val_loss /= total_val
        if cumulative_val_loss == 0:
            break

    return model


def tstep(x, y, model, optimizer, criterion, device):
    z = model(x.to(device))
    loss_training = criterion(z, y.to(device))
    optimizer.zero_grad()
    loss_training.backward()
    optimizer.step()
    return loss_training.item()


def vstep(x, y, model, criterion, device):
    z = model(x.to(device))
    loss_validation = criterion(z, y.to(device))
    return loss_validation.item()
