from dataclasses import dataclass
import torch as t

@dataclass
class Step:
    state:t.Tensor
    action:int
    reward:float
    time:int

@dataclass
class StepDetail(Step):
    gain:float

@dataclass
class Episode:
    steps:list[Step]
    gamma:float

@dataclass 
class EpisodeDetail:
    steps:list[StepDetail]
    gamma:float

