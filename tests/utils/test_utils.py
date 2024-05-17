from refl.utils import sampleFromDistribution
import torch as t

class TestSampleFromDistribution:
    def test_1(self):
        probs = t.tensor([1, 0, 0])
        assert sampleFromDistribution(probs) == 0

    def test_2(self):
        probs = t.tensor([0.5, 0.5, 0])
        assert sampleFromDistribution(probs) != 2

    def test_3(self):
        probs = t.tensor([0.0500, 0.1000, 0.0500, 0.0000, 0.0000, 0.0000, 0.8000])
        assert sampleFromDistribution(probs) in [0,1,2,6]