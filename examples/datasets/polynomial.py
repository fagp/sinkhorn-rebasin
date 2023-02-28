import torch
from torch.utils.data import Dataset
import numpy as np


class Polynomial(Dataset):
    def __init__(self, domain, std=0.05, length=100) -> None:
        super().__init__()
        self.domain = domain
        self.noise_std = std
        self.length = length

    def __str__(self) -> str:
        return "Polynomial"

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = (
            torch.tensor([[float(index) / (self.length - 1)]])
            * (self.domain[1] - self.domain[0])
            + self.domain[0]
        )
        noise = torch.randn((1, 1)) * self.noise_std
        y = self.polynomial(x) + noise
        return x, y

    def __iter__(self):
        for i in range(self.length):
            yield self[i]

    def ffcv_writer(self):
        return {}

    def ffcv_loader(self):
        return {}

    def polynomial(self, x):
        return x


class DPolynomialTask1(Polynomial):
    def __init__(self, std=0.05, length=100):
        super().__init__(domain=[-4.0, -2.0], std=std, length=2)

    def polynomial(self, x):
        return x

    def __str__(self) -> str:
        return "DPolynomialTask1"


class PolynomialTask1(Polynomial):
    def __init__(self, std=0.05, length=100):
        super().__init__(domain=[-4.0, -2.0], std=std, length=length)

    def polynomial(self, x):
        return x + 3

    def __str__(self) -> str:
        return "PolynomialTask1"


class PolynomialTask2(Polynomial):
    def __init__(self, std=0.05, length=100):
        super().__init__(domain=[-1.0, 1.0], std=std, length=length)

    def polynomial(self, x):
        return 2 * x * x - 1

    def __str__(self) -> str:
        return "PolynomialTask2"


class PolynomialTask3(Polynomial):
    def __init__(self, std=0.05, length=100):
        super().__init__(domain=[2.0, 4.0], std=std, length=length)

    def polynomial(self, x):
        return (x - 3) ** 3

    def __str__(self) -> str:
        return "PolynomialTask3"


def visualization(dataset, logger, model=None, device=torch.device("cpu"), name=""):
    for x, y in dataset:
        logger.plot(
            metric_name="polynomial",
            legend="GT_{}".format(name),
            x=x.squeeze(),
            y=y.squeeze(),
            dash=np.array(["solid"]),
        )
        if model is not None:
            model.to(device)
            with torch.no_grad():
                y_pred = model(x.float().to(device))
            logger.plot(
                metric_name="polynomial",
                legend="Pred_{}".format(name),
                x=x.squeeze(),
                y=y_pred.squeeze(),
                dash=np.array(["dash"]),
            )
