from refl.agents.agent import Agent
from refl.agents.runner import getDetailedEpisode
from refl.utils.utils import sampleFromDistribution
from refl.utils.network import MLP
from refl.utils.episode import Episode, EpisodeDetail
from refl.envs import Env
import torch
from collections import deque

class ReinforceAgent(Agent):
    def __init__(self, n_state_dims, n_latent_dims, n_actions, gamma):
        self.n_actions = n_actions
        self.policy_network = MLP(n_state_dims, n_latent_dims, n_actions)
        # self.eps = eps
        self.gamma = gamma
        self.sampler = torch.distributions.uniform.Uniform(0, 1)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-2)

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

    def update(self):
        G = 0
        policy_loss = []
        returns = deque()
        for r in self.policy_network.rewards[::-1]:
            G = r + self.gamma * G
            returns.appendleft(G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-1)
        t = 0
        for log_prob, G in zip(self.policy_network.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
            t += 1
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy_network.rewards[:]
        del self.policy_network.saved_log_probs[:]
        return policy_loss.item()

    def learn(self, env:Env, n_episodes:int, target_return:float):
        running_avg_return = 0.0
        avgs = []
        for i in range(n_episodes):
            state, _ = env.reset()
            ep_return = 0
            for t in range(1, 10000):  # Don't infinite loop while learning
                action = self.getAction(state)
                state, reward, done, _, _ = env.step(action)
                self.policy_network.rewards.append(reward)
                ep_return += reward
                if done:
                    break
            running_avg_return = 0.01 * ep_return + (1 - 0.01) * running_avg_return
            loss = self.update()
            avgs.append({'Episode':i, 'AvgReturn':running_avg_return, 'Gamma':self.gamma, 'Loss':loss})
            if i % 50 == 0:
                print('Episode {}\tLast return: {:.2f}\tAverage return: {:.2f}'.format(i, ep_return, running_avg_return))
            if running_avg_return > target_return:
                print("Solved! Running return is now {} and "
                    "the last episode return = {}".format(running_avg_return, ep_return))
                break

        return avgs 

    def evaluate(self, env:Env, n_episodes:int):
        self.learning = False
        tot_gain = 0.0
        for i in range(n_episodes):
            episode = self.generate_episode(env)
            episodeDetailed = getDetailedEpisode(episode)
            tot_gain += episodeDetailed.steps[0].gain
            if tot_gain == 0.0:
                print(episodeDetailed.steps)
        print(f'Avg gain : {tot_gain / n_episodes}')
        self.learning = True
            