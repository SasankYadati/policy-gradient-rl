from refl.agents import Agent
from refl.envs import Env
from refl.utils import Episode, EpisodeDetail, Step, StepDetail
import torch as t

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