from refl.agents.agent import Agent
from refl.agents.runner import getDetailedEpisode
from refl.utils.utils import sampleFromDistribution
from refl.utils.network import MLP
from refl.utils.episode import Episode
from refl.envs import Env
import torch

class ReinforceAgent(Agent):
    def __init__(self, n_state_dims, n_latent_dims, n_actions, eps, gamma):
        self.n_actions = n_actions
        self.policy_network = MLP(n_state_dims, n_latent_dims, n_actions)
        self.eps = eps
        self.gamma = gamma
        self.sampler = torch.distributions.uniform.Uniform(0, 1)

    def getAction(self, state):
        if isinstance(state, int) or isinstance(state, float):
            state = torch.tensor([state], dtype=torch.float32)
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = torch.unsqueeze(state, 0)
        action_logits = self.policy_network(state)
        action_probs = action_logits.softmax(-1).squeeze()
        if self.sampler.sample((1,)).item() > self.eps:
            return action_probs.argmax().item()
        else:
            return sampleFromDistribution(torch.ones(self.n_actions))

    def prepareDataForEpisode(self, episode:Episode):
        pass

    def learn(self, env:Env):
        episode, gain = self.generate_episode(env)
        detailedEpisode = getDetailedEpisode(episode)
        t = 0
        for (state, action, rew) in episode:
            pass

    def compute_gradient(self):
        pass
