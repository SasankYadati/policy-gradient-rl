from refl.agents import Agent
from refl.envs import Env
from refl.utils import Episode, EpisodeDetail, Step, StepDetail
import torch as t

def getDetailedEpisode(episode:Episode) -> EpisodeDetail:
        lastStep = episode.steps[-1]
        T = len(episode.steps)-1
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

def run_episode(agent:Agent, env:Env) -> float:
    state, rew = env.reset()
    tot_return = rew
    done = False
    while not done:
        action = agent.getAction(state)
        next_state, rew, done = env.step(action)
        tot_return += rew
        state = next_state
    return tot_return

def run_trial(agent:Agent, env:Env, n_episodes: int) -> list[float]:
    returns = []
    for _ in range(n_episodes):
        returns.append(run_episode(agent, env))
    return returns

def run_exp(agent:Agent, env:Env, n_trials:int, n_episodes:int) -> tuple[list[float], list[float]]:
    avg_returns = []
    std_returns = []
    for _ in range(n_trials):
        returns = run_trial(agent, env, n_episodes)
        std, mean = t.std_mean(t.tensor(returns))
        avg_returns.append(mean.item())
        std_returns.append(std.item())
    return avg_returns, std_returns