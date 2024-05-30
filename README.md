# policy-gradient-rl
Note - I was not able to get the algorithms to learn with gamma term in the update rule, so I have not included it.

While learning, I have considered Gridworld as solved if it gets 9.5 average return (running avg) and Cartpole as solved if it gets 500 average return.

After learning is done, all agents are evaluated on 10 episodes.

The function approximators and hyperparameter details are [here](/function_approximators_and_hyperparams.pdf)
## Reinforce

### Gridworld
All agents achieved similar and near-optimal performance.
#### Learning - Avg return over No. of Episodes
![learning curves for gridworld](/reinforce-gridworld-learning-curve.png)

#### Evaluation - Return for each episode after learning
![final evaluation](/reinforce-gridworld-eval.png)

### Cartpole
Agent with $\gamma = 1.0$ has performed the best, achieved 9000 return.
#### Learning - Avg return over No. of Episodes
![learning curves for cartpole](/reinforce-cartpole-learning-curve.png)

#### Evaluation - Return for each episode after learning
![final evaluation](/reinforce-cartpole-eval.png)

## Baseline Reinforce

### Gridworld
Agents with $\gamma=1.0, 0.99$ have learnt near-optimal policies whereas agent with $\gamma=0.95$ went on a spiral and seems to have got stuck in a local minima of value 0.
#### Learning - Avg return over No. of Episodes
![learning curves for gridworld](/baseline-reinforce-gridworld-learning-curve.png)

#### Evaluation - Return for each episode after learning
![final evaluation](/baseline-reinforce-gridworld-eval.png)

### Cartpole
Agent with $\gamma = 0.99$ has learnt quickly and done extremely well (>30k return in one of the episodes) in evaluation.
#### Learning - Avg return over No. of Episodes
![learning curves for cartpole](/baseline-reinforce-cartpole-learning-curve.png)

#### Evaluation - Return for each episode after learning
![final evaluation](/baseline-reinforce-cartpole-eval.png)
