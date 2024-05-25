from refl.agents.agent import Agent
from refl.utils.network import PolicyNet
from refl.envs import Env
import torch
from collections import deque

class ReinforceAgent(Agent):
    def __init__(self, n_state_dims, n_latent_dims, n_actions, gamma):
        self.n_actions = n_actions
        self.policy_network = PolicyNet(n_state_dims, n_latent_dims, n_actions)
        self.gamma = gamma
        self.sampler = torch.distributions.uniform.Uniform(0, 1)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-2)

    def update(self) -> float:
        G = 0
        returns = deque()
        for r in self.policy_network.rewards[::-1]:
            G = r + self.gamma * G
            returns.appendleft(G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-1)

        policy_loss = []
        for log_prob, G in zip(self.policy_network.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.cat(policy_loss).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy_network.rewards[:]
        del self.policy_network.saved_log_probs[:]
        return policy_loss.item()

    def learn(self, env:Env, n_episodes:int, target_return:float) -> list[float]:
        running_avg_return = 0.0
        avgs = []
        for i in range(n_episodes):
            state, _ = env.reset(seed=0)
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
                print("Solved! Running return is now {} and the last episode return = {}".format(running_avg_return, ep_return))
                break

        return avgs 
            