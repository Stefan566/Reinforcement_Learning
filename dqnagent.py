import numpy as np 
import random
from collections import deque
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam

class DQNagent:

    def __init__(self, state_size, action_size):
        self.learning_rate = 0.001
        self.action_size = action_size
        self.state_size = state_size

        self.eps = 1.0

        self.memory = deque(maxlen=2000)
        self.model = self.create_model(self.state_size, self.action_size, self.learning_rate)
        self.target_model = self.create_model(self.state_size, self.action_size, self.learning_rate)
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self, ss, acts, lr):
        model = Sequential()
        model.add(Dense(24,input_dim=ss, activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(acts, activation='linear'))
        # model.add(Dense(activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=lr))

        return model 

    def act(self, st):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        act_values = self.model.predict(st)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, bs):
        gamma = 0.95
        eps_min = 0.01
        eps_decay = 0.995

        mb = random.sample(self.memory, bs)
        for state, action, reward, next_state, done in mb:
            target = reward
            if not done:
                target = reward + gamma * np.amax(self.model.predict(next_state)[0])
            target2 = self.model.predict(state)
            target2[0][action] = target
            self.model.fit(state, target2, epochs=1, verbose=0)
        if self.eps > eps_min:
            self.eps *= eps_decay
