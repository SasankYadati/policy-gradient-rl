from refl.agents import Agent
from refl.utils import sampleFromDistribution, MLP, Step, StepDetail, Episode, EpisodeDetail
import torch

class BaselineReinforceAgent(Agent):
    def __init__(self, n_state_dims, n_latent_dims, n_actions, eps, gamma):
        self.n_actions = n_actions
        self.policy_network = MLP(n_state_dims, n_latent_dims, n_actions)
        self.value_network = MLP(n_state_dims, n_latent_dims, 1)
        self.eps = eps
        self.gamma = gamma
        self.sampler = torch.distributions.uniform.Uniform(0, 1)

    def getAction(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = torch.unsqueeze(state, 0)
        action_logits = self.policy_network(state)
        action_probs = action_logits.softmax(-1).squeeze()
        if self.sampler.sample((1,)).item() > self.eps:
            return action_probs.argmax().item()
        else:
            return sampleFromDistribution(torch.ones(self.n_actions))

    def prepareDataForEpisode(self, episode):
        pass

    def learn(self):
        pass

    def compute_gradient(self):
        pass
