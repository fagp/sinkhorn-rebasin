import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
import numpy as np


# Sinkhorn differentiation from https://github.com/marvin-eisenberger/implicit-sinkhorn
class Sinkhorn(torch.autograd.Function):
    """
    An implementation of a Sinkhorn layer with our custom backward module, based on implicit differentiation
    :param c: input cost matrix, size [*,m,n], where * are arbitrarily many batch dimensions
    :param a: first input marginal, size [*,m]
    :param b: second input marginal, size [*,n]
    :param num_sink: number of Sinkhorn iterations
    :param lambd_sink: entropy regularization weight
    :return: optimized soft permutation matrix
    """

    @staticmethod
    def forward(ctx, c, a, b, num_sink, lambd_sink):
        log_p = -c / lambd_sink
        log_a = torch.log(a).unsqueeze(dim=-1)
        log_b = torch.log(b).unsqueeze(dim=-2)
        for _ in range(num_sink):
            log_p -= torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b
            log_p -= torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a
        p = torch.exp(log_p)

        ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        ctx.lambd_sink = lambd_sink
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors

        device = grad_p.device

        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        grad_p *= -1 / ctx.lambd_sink * p
        K = torch.cat(
            (
                torch.cat((torch.diag_embed(a), p), dim=-1),
                torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1),
            ),
            dim=-2,
        )[..., :-1, :-1]
        t = torch.cat(
            (grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1
        ).unsqueeze(-1)
        grad_ab = torch.linalg.solve(K, t)
        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat(
            (
                grad_ab[..., m:, :],
                torch.zeros(batch_shape + [1, 1], device=device, dtype=torch.float32),
            ),
            dim=-2,
        )
        U = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
        return grad_p, grad_a, grad_b, None, None, None


def matching(alpha, **kwargs):
    # Negate the probability matrix to serve as cost matrix. This function
    # yields two lists, the row and colum indices for all entries in the
    # permutation matrix we should set to 1.
    row, col = linear_sum_assignment(-alpha, **kwargs)

    # Create the permutation matrix.
    permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return torch.from_numpy(permutation_matrix)
