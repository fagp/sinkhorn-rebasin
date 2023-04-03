import torch

try:
    from ffcv.writer import DatasetWriter
    from ffcv.loader import Loader, OrderOption

    FCCV_AVAILABLE = True
except ImportError:
    FCCV_AVAILABLE = False
import os


def dataloader(dataset=None, loader="ffcv", **opts):
    """dataloader

    Args:
        dataset (torch.utils.data.Dataset, optional): PyTorch dataset. Defaults to None.
        loader (str, optional): Loader type ['ffcv', 'torch']. Defaults to "ffcv".
        file_name (str, optional): File name for ffcv format. Ignored for torch. Defaults to data/dataset_str.beton.
        force_write (bool, optional): Force writing for ffcv format. Ignored for torch. Defaults to False.
        shuffle (bool, optional): Shuffle data. Defaults to False.
        num_workers (int, optional): Number of workers.
        batch_size (int, optional): Batch size.

    Returns:
        [torch.utils.data.DataLoader, ffcv.loader.Loader]: Data loader, either torch or ffcv.
    """
    assert loader in ["ffcv", "torch"]

    if loader == "torch":
        if "file_name" in opts:
            file_name = opts.pop("file_name")
        if "force_write" in opts:
            force = opts.pop("force_write")
        if "indices" in opts:
            opts.pop("indices")
        dl = torch.utils.data.DataLoader(dataset, **opts)
        dl.indices = torch.arange(0, len(dataset))
        return dl

    elif loader == "ffcv" and FCCV_AVAILABLE:
        file_name = "{:}/.{:}.beton".format("data", str(dataset))
        if "file_name" in opts:
            file_name = opts.pop("file_name")

        force = False
        if "force_write" in opts:
            force = opts.pop("force_write")

        if dataset is not None and (force or not os.path.exists(file_name)):
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            writer = DatasetWriter(
                file_name,
                dataset.ffcv_writer(),
                num_workers=16,
            )
            writer.from_indexed_dataset(dataset)

        order = OrderOption.SEQUENTIAL
        if "shuffle" in opts:
            shuffle = opts.pop("shuffle")
            order = OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL

        if "batch_size" not in opts:
            opts["batch_size"] = 1

        if "load_pipeline" in opts:
            load_pipeline = opts.pop("load_pipeline")
        elif dataset is not None:
            load_pipeline = dataset.ffcv_loader()
        else:
            raise ValueError("No load pipeline specified")

        loader = Loader(file_name, order=order, pipelines=load_pipeline, **opts)
        return loader
