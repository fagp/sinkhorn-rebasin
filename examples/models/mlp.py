import torch


class MLP(torch.nn.Module):
    def __init__(self, num_hidden=1):
        super().__init__()
        self.num_hidden = int(num_hidden)

        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=10, bias=True),
            torch.nn.Tanh(),
        )
        for _ in range(self.num_hidden - 1):
            self.hidden.append(
                torch.nn.Linear(in_features=10, out_features=10, bias=True)
            )
            self.hidden.append(torch.nn.Tanh())

        self.output = torch.nn.Linear(in_features=10, out_features=1, bias=True)

    def forward(self, x):
        x = self.hidden(x.float())
        x = self.output(x)
        return x


class MLP2(torch.nn.Module):
    def __init__(self, num_hidden=1, num_inputs=1, num_classes=10, dropout_p=0.0):
        super().__init__()
        self.num_hidden = num_hidden = int(num_hidden)
        self.num_classes = num_classes = int(num_classes)
        self.num_inputs = num_inputs = int(num_inputs)
        self.dropout_p = float(dropout_p)
        self.num_features = 512

        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.num_inputs, out_features=self.num_features, bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p),
        )
        for _ in range(num_hidden - 1):
            self.linear1.append(
                torch.nn.Linear(
                    in_features=self.num_features,
                    out_features=self.num_features,
                    bias=True,
                )
            )
            self.linear1.append(torch.nn.ReLU())
            self.linear1.append(torch.nn.Dropout(self.dropout_p))

        self.classification = torch.nn.Linear(
            in_features=self.num_features, out_features=self.num_classes, bias=True
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x.float())
        x = self.classification(x)
        return x
