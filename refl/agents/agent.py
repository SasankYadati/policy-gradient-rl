from refl.envs.base_env import Env
from refl.utils.episode import Episode, Step

class Agent:
    gamma:float
    def __init__():
        pass

    def getAction(self, state):
        pass

    def generate_episode(self, env:Env) -> Episode:
        state, rew = env.reset()
        steps = []
        rewards = []
        done = False
        t = 0
        while not done:
            action = self.getAction(state)
            next_state, rew, done = env.step(action)
            print(f's={state}, a={action}, r={rew}, s`={next_state}')
            steps.append(Step(state, action, rew, t))
            rewards.append(rew)
            state = next_state
            t += 1
        
        return Episode(steps, self.gamma)