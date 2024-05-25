from refl.agents import Agent
from refl.utils import sampleFromDistribution, PolicyNet, ValueNet, Step, StepDetail, Episode, EpisodeDetail
from collections import deque
import torch

class BaselineReinforceAgent(Agent):
    def __init__(self, n_state_dims, n_latent_dims, n_actions, gamma):
        self.n_actions = n_actions
        self.policy_network = PolicyNet(n_state_dims, n_latent_dims, n_actions)
        self.value_network = ValueNet(n_state_dims, n_latent_dims)
        self.gamma = gamma
        self.sampler = torch.distributions.uniform.Uniform(0, 1)
        self.optimizer_policy = torch.optim.Adam(self.policy_network.parameters(), lr=1e-2)
        self.optimizer_value = torch.optim.Adam(self.value_network.parameters(), lr=1e-2)

    def storeStateValue(self, state):
        if isinstance(state, torch.Tensor):
            state = torch.unsqueeze(state, 0)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
        value = self.value_network(state)
        self.value_network.values.append(value)
        return value.item()

    def learn(self, env, n_episodes, target_return):
        running_avg_return = 0.0
        avgs = []
        for i in range(n_episodes):
            state, _ = env.reset(seed=0)
            ep_return = 0
            for t in range(1, 10000):  # Don't infinite loop while learning
                action = self.getAction(state)
                self.storeStateValue(state)
                state, reward, done, _, _ = env.step(action)
                self.policy_network.rewards.append(reward)
                ep_return += reward
                if done:
                    # self.value_network.values.append(torch.tensor(0.0)) # value of terminal state is 0
                    break
            running_avg_return = 0.01 * ep_return + (1 - 0.01) * running_avg_return
            critic_loss, policy_loss = self.update()
            avgs.append({'Episode':i, 'AvgReturn':running_avg_return, 'Gamma':self.gamma, 'Critic Loss':critic_loss, 'Policy Loss': policy_loss})
            if i % 50 == 0:
                print('Episode {}\tLast return: {:.2f}\tAverage return: {:.2f}'.format(i, ep_return, running_avg_return))
            if running_avg_return > target_return:
                print("Solved! Running return is now {} and the last episode return = {}".format(running_avg_return, ep_return))
                break

        return avgs 

    def update(self):
        critic_loss = self.update_critic()
        policy_loss = self.update_policy()
        del self.policy_network.rewards[:]
        del self.policy_network.saved_log_probs[:]
        del self.value_network.values[:]
        return critic_loss, policy_loss

    def update_policy(self):        
        returns = self.getReturns()

        policy_loss = []
        for log_prob, G, V in zip(self.policy_network.saved_log_probs, returns, self.value_network.values):
            policy_loss.append(-log_prob * (G.item()-V.item()))
        
        policy_loss = torch.cat(policy_loss).sum()
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()
        return policy_loss.item()

    def update_critic(self):
        returns = self.getReturns()
        T = len(self.value_network.values) - 1
        
        critic_losses = []
        for t in range(0, T):
            delta = (returns[t].item() - self.value_network.values[t].item())
            critic_loss = delta * -self.value_network.values[t] 
            critic_losses.append(critic_loss)
        self.optimizer_value.zero_grad()
        critic_losses = torch.cat(critic_losses).sum()
        critic_losses.backward(retain_graph=True)
        self.optimizer_value.step()
        return critic_losses.item()
    
    def getReturns(self):
        returns = deque()
        G = 0
        for r in self.policy_network.rewards[::-1]:
            G = r + self.gamma * G
            returns.appendleft(G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-1)
        return returns