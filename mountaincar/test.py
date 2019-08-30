import gym
import sys
import time
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import itertools

def build_model(env):
	model = Sequential()
	model.add(Dense(100, input_dim = 2, activation = "relu"))
	model.add(Dense(50, activation = "relu"))
	model.add(Dense(env.action_space.n, activation = "linear"))
	model.compile(loss = "mse", optimizer = Adam(lr = 0.001))
	return model

def test(env, model, samples):
	print("==========\nTest\n==========")
	total_reward = 0
	time.sleep(1)
	for i in range(samples):
		print("Sample {}/{}".format(i+1, samples))
		time.sleep(0.5)
		state = env.reset()
		for t in itertools.count():
			state = state.reshape(1,2)
			env.render()
			time.sleep(0.03)
			state = state.reshape(1,2)
			action = np.argmax(model.predict(state, verbose = 0))
			next_state, reward, done, _ = env.step(action)
			total_reward += reward
			if done:
				print("Total Reward : {}".format(total_reward))
				total_reward = 0
				break
			state = next_state

def test_performance(env, model, num_episodes = 100):
	rewards = []
	for i in range(num_episodes):
		state = env.reset()
		total_reward = 0
		for t in itertools.count():
			state = state.reshape(1,2)
			action = np.argmax(model.predict(state, verbose = 0))
			next_state, reward, done, _ = env.step(action)
			total_reward += reward
			if done:
				rewards.append(total_reward)
				total_reward = 0
				break
			state = next_state

	print("Average Reward {}".format(np.sum(rewards) * 1.0 / num_episodes))

env = gym.make("MountainCar-v0")

#model = build_model(env)

#model.load_weights("MountainCar_20000.h5")

model = load_model("MountainCar.h5")

num_samples = 10

test_performance(env, model)

test(env, model, num_samples)