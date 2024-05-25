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

def getDetailedEpisode(episode:Episode) -> EpisodeDetail:
        lastStep = episode.steps[-1]
        T = len(episode.steps)
        lastDetailedStep = StepDetail(lastStep.state, lastStep.action, lastStep.reward, T, 0.0)
        detailedSteps = [lastDetailedStep]
        for step in reversed(episode.steps[:-1]):
            prevDetailedStep = getPrevStepDetail(step, detailedSteps[-1], episode.gamma)
            detailedSteps.append(prevDetailedStep)
        
        return EpisodeDetail(list(reversed(detailedSteps)), episode.gamma)

def getPrevStepDetail(step:Step, nextStepDetail:StepDetail, gamma) -> StepDetail:
    gain = step.reward + gamma * nextStepDetail.gain
    time = nextStepDetail.time - 1
    return StepDetail(step.state, step.action, step.reward, time, gain)

