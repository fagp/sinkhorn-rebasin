import numpy as np
from matplotlib import cm
from torchvision import utils


def visualize_kernels(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    tensor = tensor.detach().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(1)
    n, c, w, h = tensor.shape

    grayscale = False

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)
        grayscale = True

    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    grid = grid.numpy().transpose((1, 2, 0))

    if grayscale:
        grid = grid[:, :, 0]
        grid = cm.Spectral_r(grid)[..., :3]
    return grid
