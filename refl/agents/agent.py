from refl.envs.base_env import Env
from refl.utils import Episode, Step, getDetailedEpisode
import torch

class Agent:
    gamma:float
    policy_network:torch.nn.Module
    def __init__():
        pass

    def getAction(self, state):
        if isinstance(state, torch.Tensor):
            state = torch.unsqueeze(state, 0)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.policy_network.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def generate_episode(self, env:Env) -> Episode:
        state, rew = env.reset(seed=0)
        steps = []
        rewards = []
        done = False
        t = 0
        gain = 0.0
        while not done:
            action = self.getAction(state)
            next_state, rew, done, _, _ = env.step(action)
            steps.append(Step(state, action, rew, t))
            gain += rew
            state = next_state
            t += 1
        
        return Episode(steps, self.gamma), gain

    def evaluate(self, env:Env, n_episodes:int):
        returns = []
        for _ in range(n_episodes):
            _, gain = self.generate_episode(env)
            returns.append(gain)
        print(f'Avg gain : {sum(returns) / n_episodes}')
        return returns