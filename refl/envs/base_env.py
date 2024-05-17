from torch import Tensor

class Env:
    n_state_dims:int
    n_actions:int
    def reset(self) -> Tensor:
        pass

    def step(self, action:int) -> Tensor:
        pass