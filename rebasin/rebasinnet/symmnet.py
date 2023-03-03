import torch
from copy import deepcopy
from .sinkhorn import Sinkhorn, matching
from .graph.auto_graph import solve_graph

class ReparamNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.output = deepcopy(model)
        self.model = deepcopy(model)
        for p1, p2 in zip(self.model.parameters(), self.output.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

    def set_model(self, model):
        self.model = deepcopy(model)
        for p1 in self.model.parameters():
            p1.requires_grad = False

    def forward(self, P):
        for p1, p2 in zip(self.output.parameters(), self.model.parameters()):
            p1.data = p2.data.clone()

        for p1 in self.output.parameters():
            p1._grad_fn = None

        for (name, p1), p2 in zip(
            self.output.named_parameters(), self.model.parameters()
        ):
            if (
                name not in self.map_param_index
                or name not in self.map_prev_param_index
            ):
                continue
            i = self.perm_dict[self.map_param_index[name]]
            j = self.perm_dict[self.map_prev_param_index[name]] if self.map_prev_param_index[name] is not None else None

            if "bias" in name:
                if i is not None :
                    p1.copy_(P[i] @ p2)
                else:
                    continue

            # batchnorm
            elif len(p1.shape) == 1:
                if i is not None:
                    p1.copy_((P[i] @ p2.view(p1.shape[0], -1)).view(p2.shape))

            # mlp / cnn
            elif "weight" in name:
                if i is not None and j is None:
                    p1.copy_((P[i] @ p2.view(P[i].shape[0], -1)).view(p2.shape))

                if i is not None and j is not None:
                    p1.copy_(
                        (
                            P[j].view(1, *P[j].shape)
                            @ (P[i] @ p2.view(P[i].shape[0], -1)).view(
                                p2.shape[0], P[j].shape[0], -1
                            )
                        ).view(p2.shape)
                    )

                if i is None and j is not None:
                    p1.copy_(
                        (
                            P[j].view(1, *P[j].shape)
                            @ p2.view(p2.shape[0], P[j].shape[0], -1)
                        ).view(p2.shape)
                    )

        return self.output

    def to(self, device):
        self.output.to(device)
        self.model.to(device)

        return self
    
class RebasinNet(torch.nn.Module):
    def __init__(
        self,
        model,
        input_shape,
        remove_nodes = list(),
        l=1.0,
        tau=1.0,
        n_iter=20,
        operator="implicit",
    ):
        super().__init__()
        assert operator in [
            "implicit",
        ], "Operator must be either `implicit`"

        self.reparamnet = ReparamNet(model)
        input = torch.randn(input_shape)
        perm_dict,n_perm,permutation_g,parameter_map = solve_graph(model,input,remove_nodes=remove_nodes)
        
        P_sizes = [None] * n_perm
        map_param_index = dict()
        map_prev_param_index = dict()
        nodes = list(permutation_g.nodes.keys())
        for name, p in model.named_parameters():
            if parameter_map[name] not in nodes:
                continue
            else:
                map_param_index[name] = permutation_g.naming[parameter_map[name]]
            parents = permutation_g.parents(parameter_map[name])
            map_prev_param_index[name] = (
                None if len(parents) == 0 else permutation_g.naming[parents[0]]
                )

            if "weight" in name:
                if len(p.shape) == 1:  # batchnorm
                    pass  # no permutation : bn is "part" for the previous one like biais
                else:
                    if map_param_index[name] is not None and perm_dict[map_param_index[name]] is not None:
                        perm_index = perm_dict[map_param_index[name]]
                        P_sizes[perm_index] = (p.shape[0], p.shape[0])

        self.reparamnet.map_param_index = map_param_index
        self.reparamnet.map_prev_param_index = map_prev_param_index
        self.reparamnet.perm_dict = perm_dict

        self.p = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.eye(ps[0]) + torch.randn(ps) * 0.1, requires_grad=True
                )
                if ps is not None
                else None
                for ps in P_sizes
            ]
        )

        self.l = l
        self.tau = tau
        self.n_iter = n_iter
        self.operator = operator

    def random_init(self):
        for p in self.p:
            ci = torch.randperm(p.shape[0])
            p.data = (torch.eye(p.shape[0])[ci, :]).to(p.data.device)

    def identity_init(self):
        for p in self.p:
            p.data = torch.eye(p.shape[0]).to(p.data.device)

    def forward(self, x=None):

        if self.training:
            gk = list()
            for i in range(len(self.p)):
                if self.operator == "implicit":
                    sk = Sinkhorn.apply(
                        -self.p[i] * self.l,
                        torch.ones((self.p[i].shape[0])).to(self.p[0].device),
                        torch.ones((self.p[i].shape[1])).to(self.p[0].device),
                        self.n_iter,
                        self.tau,
                    )

                gk.append(sk)

        else:
            gk = [
                matching(p.cpu().detach().numpy()).float().to(self.p[0].device)
                for p in self.p
            ]

        m = self.reparamnet(gk)
        if x is not None and x.ndim == 1:
            x.unsqueeze_(0)

        if x is not None:
            return m(x)

        return m

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.reparamnet.output.zero_grad(set_to_none)
        return super().zero_grad(set_to_none)

    def parameters(self, recurse: bool = True):
        return self.p.parameters(recurse)

    def to(self, device):
        for p in self.p:
            if p is not None:
                p.data = p.data.to(device)

        return self
