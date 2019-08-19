import gym
import sys
import time
import random
import numpy as np
import itertools
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def nth_root(num, n):
	return(n**(1/num))

class DeepQLearning:
	
	def __init__(self, env):
		#init 
		self.memory = []
		self.gamma = 0.99
		self.epsilon = 0.1
		self.learning_rate = 0.001
		self.batch_size = 20
		self.action_space = env.action_space.n

		self.model = Sequential()
		self.model.add(Dense(64, input_dim = env.observation_space.shape[0], activation = "relu"))
		self.model.add(Dense(32, activation = "relu"))
		self.model.add(Dense(self.action_space, activation = "linear"))
		self.model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
	
	def store_transition(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def epsilon_greedy_policy(self, state):
		A = np.ones(self.action_space)*self.epsilon/self.action_space
		best_action = np.argmax(self.model.predict(state))
		A[best_action] += 1 - self.epsilon
		action = np.random.choice(range(len(A)), p = A)
		return action

	def experience_replay(self):
		batch = random.choices(self.memory, k = self.batch_size)
		for state, action, reward, next_state, done in batch:
			q_update = reward
			if not done:
				q_update += self.gamma*(np.max(self.model.predict(next_state)[0]))
			q_values = self.model.predict(state)
			q_values[0][action] = q_update
			self.model.fit(state, q_values, verbose = 0)

num_train_episodes = 30000
env = gym.make("MountainCar-v0")
dql = DeepQLearning(env)
for i_episode in range(num_train_episodes):
	state = env.reset()
	state = state.reshape(1, env.observation_space.shape[0])
	score = 0
	for t in itertools.count():
		action = dql.epsilon_greedy_policy(state)
		next_state, reward, done, _ = env.step(action)
		reward = reward if not done else -reward
		score += reward
		next_state = next_state.reshape(1, env.observation_space.shape[0])
		dql.store_transition(state, action, reward, next_state, done)
		state = next_state
		if done:
			print("Episode {} reward {}".format(i_episode+1, score))
			break
		dql.experience_replay()

dql.model.save("MountainCar_EP.h5")
