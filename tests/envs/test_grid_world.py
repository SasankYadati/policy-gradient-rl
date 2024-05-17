from refl.envs import GridWorldEnv, getStateID, RIGHT, UP

class TestGridWorld:
    gw = GridWorldEnv()
    def test_transition_1(self):
        probs = self.gw.getTransitionProbs(getStateID(1,1), RIGHT)
        assert probs.sum() == 1.0
        assert probs[getStateID(2,1)] == 0.8
        assert probs[getStateID(1,0)] == 0.05
        assert probs[getStateID(1,2)] == 0.05
        assert probs[getStateID(1,1)] == 0.1
    
    def test_transition_2(self):
        probs = self.gw.getTransitionProbs(getStateID(1,2), UP)
        assert probs.sum() == 1.0
        assert probs[getStateID(0,2)] == 0.05
        assert probs[getStateID(1,2)] == 0.15
        assert probs[getStateID(1,3)] == 0.8
    
    def test_transition_3(self):
        probs = self.gw.getTransitionProbs(getStateID(0,0), UP)
        assert probs.sum() == 1.0
        assert probs[getStateID(0,0)] == 0.15
        assert probs[getStateID(0,1)] == 0.8
        assert probs[getStateID(1,0)] == 0.05
    
    def test_transition_4(self):
        T = self.gw.getTransistionFn()
        assert T[getStateID(1,1)][RIGHT].sum() == 1.0
        assert T[getStateID(1,1)][RIGHT][getStateID(2,1)] == 0.8
        assert T[getStateID(1,1)][RIGHT][getStateID(1,0)] == 0.05
        assert T[getStateID(1,1)][RIGHT][getStateID(1,2)] == 0.05
        assert T[getStateID(1,1)][RIGHT][getStateID(1,1)] == 0.1