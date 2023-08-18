import torch
import torch.nn as nn


class DistCosineLoss(nn.Module):
    """
    Suitable for neurons aligment
    """

    def __init__(self, modela=None):
        super(DistCosineLoss, self).__init__()
        self.modela = modela
        self.eps = torch.tensor(1e-8)
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb):
        loss = 0
        num_params = 0
        for p1, p2 in zip(self.modela.parameters(), modelb.parameters()):
            num_params += p1.numel()
            loss += (
                2
                - (
                    (p1 * p2).sum()
                    / torch.max(
                        torch.linalg.vector_norm(p1) * torch.linalg.vector_norm(p2),
                        self.eps,
                    )
                )
                + 1
            )

        loss /= num_params

        return loss


class DistL2Loss(nn.Module):
    """
    Suitable for neurons aligment
    """

    def __init__(self, modela=None):
        super(DistL2Loss, self).__init__()
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb):
        loss = 0
        num_params = 0
        for p1, p2 in zip(self.modela.parameters(), modelb.parameters()):
            num_params += p1.numel()
            loss += torch.pow(p1 - p2, 2).sum()

        loss /= num_params

        return loss


class DistL1Loss(nn.Module):
    """
    Suitable for neurons aligment
    """

    def __init__(self, modela=None):
        super(DistL1Loss, self).__init__()
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb):
        loss = 0
        num_params = 0
        for p1, p2 in zip(self.modela.parameters(), modelb.parameters()):
            num_params += p1.numel()
            loss += (p1 - p2).abs().sum()

        loss /= num_params

        return loss


class MidLoss(nn.Module):
    """
    Suitable for linear mode connectivity
    """

    def __init__(self, modela=None, criterion=None):
        super(MidLoss, self).__init__()

        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()

        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb, input, target):
        mid_lambda = torch.tensor([0.5]).to(input.device)

        for p1, p2 in zip(modelb.parameters(), self.modela.parameters()):
            p1.mul_(0.5)
            p1.add_(0.5 * p2.data)

        z = modelb(input)
        loss = self.criterion(z, target)

        return loss


class RndLoss(nn.Module):
    """
    Suitable for linear mode connectivity
    """

    def __init__(self, modela=None, criterion=None):
        super(RndLoss, self).__init__()

        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()

        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb, input, target):
        random_l = torch.rand((1,)).to(input.device)

        for p1, p2 in zip(modelb.parameters(), self.modela.parameters()):
            p1.add_((random_l / (1 - random_l)) * p2.data)
            p1.mul_((1 - random_l))

        z = modelb(input)
        loss = self.criterion(z, target)

        return loss
