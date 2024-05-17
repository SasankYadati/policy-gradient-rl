import torch as t
import einops

class MLP(t.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.l1 = t.nn.Linear(n_in, n_hidden)
        self.l2 = t.nn.Linear(n_hidden, n_out)

    def forward(self, x):
        hidden = self.l1(x)
        hidden = t.nn.functional.gelu(hidden)
        out = self.l2(hidden)
        return out