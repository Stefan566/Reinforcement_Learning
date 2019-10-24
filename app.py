import numpy as np 
import random

from dqnagent import DQNagent
import gym

episodes = 1000
batch_size = 32

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n 
agent = DQNagent(state_size, action_size)
done = False

for ep in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for t in range(500):
        env.render()
        ac = agent.act(state)
        next_state, reward, done, _ = env.step(ac)
        if done:
            reward = -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, ac, reward, next_state, done)
        if done:
            print("Episode: {}/{}, Score: {}, Epsilon: {:.2}".format(ep, episodes, t, agent.eps))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)