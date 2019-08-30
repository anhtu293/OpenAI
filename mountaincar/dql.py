import gym
import sys
import time
import numpy as np
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def nth_root(num, n):
	return(n**(1/num))

#initiate model
def build_model(env):
	model = Sequential()
	model.add(Dense(100, input_dim = 2, activation = "relu"))
	model.add(Dense(50, activation = "relu"))
	model.add(Dense(env.action_space.n, activation = "linear"))
	model.compile(loss = "mse", optimizer = Adam(lr = 0.001))
	return model

def modelLoader(fname):
	model = load_model(fname)
	model.compile(loss = "mse", optimizer = Adam(lr = 0.001))
	return model

#SGD
def TD_update(model, state, Q_updated):
	model.fit(state, Q_updated, verbose = 0)
	return model

#approximation function for getting Q
def approximation_function(model, state):
	return(model.predict(state, verbose = 0))

def epsilon_greedy_policy(model, state, nA, epsilon):
	A = np.ones(nA) * epsilon / nA
	best_action = np.argmax(approximation_function(model, state))
	A[best_action] += 1 - epsilon
	action = np.random.choice(range(len(A)), p = A)
	return action

def DeepQLearning(env, model, num_episodes, gamma = 0.99, epsilon = 1):
	#create model
	#model = build_model(env)

	#epsilon decay
	final_epsilon = 0.01
	decay_epsilon = nth_root(num_episodes, final_epsilon/epsilon)

	scores = []

	print("==================Start Training=================\n")
	time.sleep(0.5)

	success = 0
	for i_episode in range(1, num_episodes):
		if i_episode % 100 == 0:
			avg_score = np.mean(np.asarray(scores))
			print("\rEpisode {}/{} : \nAverage score over last 100 episodes {} 		Success : {}".format(i_episode, num_episodes, avg_score, success))
			scores.clear()
			success = 0
		#initiate a episode
		epsilon = epsilon * decay_epsilon
		state = env.reset()
		highest_point = state[0]
		score = 0
		for t in itertools.count():
			state = state.reshape(1,2)
			action = epsilon_greedy_policy(model, state, env.action_space.n, epsilon)
			next_state, reward, done, _ = env.step(action)
			next_state = next_state.reshape(1,2)

			if next_state[0][0] > state[0][0]:
				reward += 10*np.absolute(next_state[0][0] - state[0][0])
				highest_point = next_state[0][0]

			if t < 199 and done:
				reward += 199-t

			score += reward
			
			Q_new = reward + gamma*np.max(approximation_function(model, next_state)[0])
			Q_updated = approximation_function(model, state)
			Q_updated[0][action] = Q_new
			TD_update(model, state, Q_updated)
			if done:
				scores.append(score)
				if t < 199:
					success += 1
				break
			state = next_state
		if i_episode % 20000 == 0:
			model.save_weights("MountainCar_{}.h5".format(i_episode))
			print("\n Checkpoint episode {} : Model saved !".format(i_episode))
	print("\nTraining Completed")
	return model

env = gym.make("MountainCar-v0")

num_train_episodes = 100000

model = build_model(env)

model = DeepQLearning(env, model, num_train_episodes)

model.save_weights("MountainCar_weights.h5")
model.save("MountainCar.h5")
