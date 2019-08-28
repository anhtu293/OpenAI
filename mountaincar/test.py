import gym
import sys
import time
from collections import defaultdict
from tensorflow.keras.models import load_model
import numpy as np
import itertools

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
model = load_model("MountainCar.h5")

num_samples = 10

test_performance(env, model)

test(env, model, num_samples)