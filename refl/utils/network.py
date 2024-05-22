import torch as t
import einops

class MLP(t.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(MLP, self).__init__()
        self.affine1 = t.nn.Linear(n_in, n_hidden)
        self.dropout = t.nn.Dropout(p=0.6)
        self.affine2 = t.nn.Linear(n_hidden, n_out)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = t.nn.functional.relu(x)
        action_scores = self.affine2(x)
        return t.nn.functional.softmax(action_scores, dim=1)