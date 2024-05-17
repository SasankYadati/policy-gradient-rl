import torch as t

def sampleFromDistribution(probs:t.Tensor):
    assert len(probs.shape) == 1
    probs = probs.cumsum(0)
    return (t.rand(1) > probs).sum(dim=-1).item()