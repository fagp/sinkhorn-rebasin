from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms

try:
    from ffcv.fields import FloatField, IntField, NDArrayField
    from ffcv.fields.decoders import FloatDecoder, IntDecoder, NDArrayDecoder
    from ffcv.transforms import ToTensor, Squeeze

    FCCV_AVAILABLE = True
except ImportError:
    FCCV_AVAILABLE = False
import numpy as np
from scipy.ndimage import rotate


class TorchvisionClassification(Dataset):
    def __init__(self, dataset, *args, **kwargs) -> None:
        super().__init__()
        self.dataset = dataset(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = np.array(x).astype("float32") / 255.0
        return x, y

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self[i]

    def ffcv_writer(self):
        if FCCV_AVAILABLE:
            x, y = self[0]
            return {
                "x": NDArrayField(dtype=np.dtype("float32"), shape=x.shape),
                "y": NDArrayField(dtype=np.dtype("float32"), shape=y.shape),
            }
        return {}

    def ffcv_loader(self):
        if FCCV_AVAILABLE:
            return {
                "x": [NDArrayDecoder(), ToTensor()],
                "y": [NDArrayDecoder(), ToTensor()],
            }
        return {}


class SubsetImageNetDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.dataset = datasets.ImageFolder(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = np.array(x).astype("float32")
        return x, y

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self[i]

    def ffcv_writer(self):
        if FCCV_AVAILABLE:
            x, y = self[0]
            return {
                "x": NDArrayField(dtype=np.dtype("float32"), shape=x.shape),
                "y": NDArrayField(dtype=np.dtype("float32"), shape=y.shape),
            }
        return {}

    def ffcv_loader(self):
        if FCCV_AVAILABLE:
            return {
                "x": [NDArrayDecoder(), ToTensor()],
                "y": [NDArrayDecoder(), ToTensor()],
            }
        return {}

    def __str__(self) -> str:
        return "imagenet"


class MNistDataset(TorchvisionClassification):
    def __init__(self, *args, **kwargs) -> None:
        if "angle" in kwargs:
            self.angle = kwargs.pop("angle")
        super().__init__(datasets.MNIST, *args, **kwargs)

    def __str__(self) -> str:
        return "mnist"

    def ffcv_writer(self):
        if FCCV_AVAILABLE:
            x, y = self[0]
            return {
                "x": NDArrayField(dtype=np.dtype("float32"), shape=x.shape),
                "y": IntField(),
            }
        return {}

    def ffcv_loader(self):
        if FCCV_AVAILABLE:
            return {
                "x": [NDArrayDecoder(), ToTensor()],
                "y": [IntDecoder(), ToTensor(), Squeeze()],
            }
        return {}


class SmallMNistDataset(TorchvisionClassification):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(datasets.MNIST, *args, **kwargs)

    def __len__(self):
        return 10000

    def __str__(self) -> str:
        return "mnist"

    def ffcv_writer(self):
        if FCCV_AVAILABLE:
            x, y = self[0]
            return {
                "x": NDArrayField(dtype=np.dtype("float32"), shape=x.shape),
                "y": IntField(),
            }
        return {}

    def ffcv_loader(self):
        if FCCV_AVAILABLE:
            return {
                "x": [NDArrayDecoder(), ToTensor()],
                "y": [IntDecoder(), ToTensor(), Squeeze()],
            }
        return {}


class RotatedMNistDataset(TorchvisionClassification):
    def __init__(self, *args, **kwargs) -> None:
        self.angle = 9
        if "angle" in kwargs:
            self.angle = kwargs.pop("angle")

        super().__init__(datasets.MNIST, *args, **kwargs)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = rotate(x, self.angle, (1, 0), False)
        x = np.array(x).astype("float32") / 255.0
        return x, y

    def __str__(self) -> str:
        return "rotated_mnist"


class SmallRotatedMNistDataset(TorchvisionClassification):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(datasets.MNIST, *args, **kwargs)

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = rotate(x, 45, (1, 0), False)
        x = np.array(x).astype("float32") / 255.0
        return x, y

    def __str__(self) -> str:
        return "rotated_mnist"


class CIFAR10Dataset(TorchvisionClassification):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(datasets.CIFAR10, *args, **kwargs)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = x.numpy()
        return x, np.array([y])

    def __str__(self) -> str:
        return "cifar10" + ("_train" if self.dataset.train else "_test")

    def ffcv_writer(self):
        if FCCV_AVAILABLE:
            x, y = self[0]
            return {
                "x": NDArrayField(dtype=np.dtype("float32"), shape=x.shape),
                "y": NDArrayField(dtype=np.dtype("int"), shape=y.shape),
            }
        return {}

    def ffcv_loader(self):
        if FCCV_AVAILABLE:
            return {
                "x": [NDArrayDecoder(), ToTensor()],
                "y": [NDArrayDecoder(), ToTensor(), Squeeze(1)],
            }
        return {}


class SmallCIFAR10Dataset(CIFAR10Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __len__(self):
        return 10000


class RotatedCIFAR10Dataset(TorchvisionClassification):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(datasets.CIFAR10, *args, **kwargs)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = rotate(x, 9, (1, 0), False)
        x = np.array(x).astype("float32") / 255.0
        x = x.transpose(2, 1, 0)
        return x, y

    def __str__(self) -> str:
        return "rotated_cifar10"
