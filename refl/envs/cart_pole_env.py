import math
import torch as t
from math import pi as PI
from dataclasses import dataclass
from refl.envs.base_env import Env

G = 9.8
STEP = 0.02

@dataclass
class State:
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    t: float

    def getList(self):
        return [self.x, self.x_dot, self.theta, self.theta_dot, self.t]

    def getTensor(self):
        return t.tensor(self.getList(), dtype=t.float32)

INITIAL_STATE = State(0, 0, 0, 0, 0)

F = 10 #Newton
CART_MASS = 1 #kg
POLE_MASS = 0.1 #kg

POLE_LEN = 1

X_RANGE = (-3, 3)
V_RANGE = (-10, 10)
THETA_RANGE = (-5*PI/12, 5*PI/12)
THETA_V_RANGE = (-PI, PI)
MAX_TIME = 20.0

LEFT = 0
RIGHT = 1

ACTIONS = {LEFT:-1, RIGHT:1}

REWARD = 1.0

class CartPoleEnv(Env):
    def __init__(self):
        self.g = G
        self.mc = CART_MASS
        self.mp = POLE_MASS
        self.l = POLE_LEN
        self.F = F
        self.state = INITIAL_STATE
        self.delta_t = STEP
        self.n_state_dims = 4
        self.n_actions = 2

    def isDone(self, state: State) -> bool:
        x, _, theta, _, t = state.getList()
        return (
            (x < X_RANGE[0] or x > X_RANGE[1]) or
            (theta < THETA_RANGE[0] or theta > THETA_RANGE[1]) or
            (t >= 20.0)
        )

    def clipVelocity(self, x_dot: float) -> float:
        clipped_v = min(max(x_dot, V_RANGE[0]), V_RANGE[1])
        return clipped_v
    
    def clipAngVelocity(self, theta_dot: float) -> float:
        clipped_theta_v = min(max(theta_dot, THETA_V_RANGE[0]), THETA_V_RANGE[1])
        return clipped_theta_v

    def transition_fn(self, state: State, action: int) -> State:
        assert action in ACTIONS.keys()
        x, x_dot, theta, theta_dot, t = state.getList()

        F = ACTIONS[action] * self.F
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        tmp = (F + self.mp * self.l * theta_dot ** 2 * sintheta) / (self.mp + self.mc)

        theta_dot_dot = (self.g * sintheta + costheta * (-tmp)) / (self.l * (4.0/3.0 - (self.mp * costheta ** 2)/(self.mc + self.mp)))
        x_dot_dot = tmp - (self.mp * self.l * theta_dot_dot * costheta) / (self.mc + self.mp)

        x = x + self.delta_t * x_dot
        x_dot = x_dot + self.delta_t * x_dot_dot
        x_dot = self.clipVelocity(x_dot)
        
        theta = theta + self.delta_t * theta_dot
        theta_dot = theta_dot + self.delta_t * theta_dot_dot
        theta_dot = self.clipAngVelocity(theta_dot)

        t += self.delta_t

        return State(x, x_dot, theta, theta_dot, t)

    def step(self, action: int) -> tuple[t.Tensor, float, bool]:
        self.state = self.transition_fn(self.state, action)
        is_done = self.isDone(self.state)
        return self.state.getTensor()[:-1], REWARD, is_done

    def reset(self, seed:int) -> tuple[t.Tensor, float] :
        self.state = INITIAL_STATE
        return self.state.getTensor()[:-1], REWARD