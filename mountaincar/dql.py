import gym
import sys
import time
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import itertools

#Func approx 
def build_model(env):
	model = Sequential()
	model.add(Dense(100, input_dim = 4, activation = "relu"))
	model.add(Dense(100, activation = "relu"))
	model.add(Dense(env.action_space.n, activation = "linear"))
	model.compile(loss = "mse", optimizer = Adam(lr = 0.001))
	return model

def TD_update(model, state, new_Q):
	model.fit(state, new_Q, verbose = 0)
	return model

def Q_function_approximation(model, state):
	return model.predict(state, verbose = 0)

def epsilon_greedy_policy(model, nA, epsilon, state):
	A = np.ones(nA)*epsilon/nA
	best_action = np.argmax(Q_function_approximation(model, state))
	A[best_action] += 1 - epsilon
	return A

def nth_root(num, n):
	return(n**(1/num))

def DeepQLearning(env, num_episodes, gamma = 0.99, epsilon = 1):
	#find epsilon decay
	final_epsilon = 0.01
	epsilon_decay = nth_root(num_episodes, final_epsilon/epsilon)
	#init model
	model = build_model(env)
	score_episodes = []

	print("==================\nStart Training\n==================")
	time.sleep(0.5)
	for i_episode in range(num_episodes):
		if (i_episode+1) % 100 == 0:
			total_scores = 0
			for score in score_episodes:
				total_scores += score
			avg_score = (total_scores*1.0)/len(score_episodes)
			score_episodes.clear()
			print("\r Episode {} Average scores last 100 episodes : {}".format(i_episode+1,avg_score), end = "")
			sys.stdout.flush()
		#decay epsilon
		epsilon = epsilon * epsilon_decay
		state = env.reset()

		score = 0
		state = state.reshape(1,4)
		highest_point = state[0][0]
		#print("\n++++Episode {}++++\n".format(i_episode))
		for t in itertools.count():
			#state = state.reshape(1,2)
			"""
			env.render()
			time.sleep(0.03)
			"""
			action_probs = epsilon_greedy_policy(model, env.action_space.n, epsilon, state)
			action = np.random.choice(range(len(action_probs)), p = action_probs)
			next_state, reward, done, _ = env.step(action)
			next_state = next_state.reshape(1,4)

			"""
			if next_state[0][0] > highest_point:
				reward += np.absolute(next_state[0][0] - highest_point)*10
				highest_point = next_state[0][0]
			"""

			score += reward
			"""if (t+1) % 1 == 0:
				print("Episode {} : t = {} reward = {}<".format(i_episode+1, t+1, score))
			"""
			Q_predict = Q_function_approximation(model, state)
			
			new_Q = reward + gamma * np.max(Q_function_approximation(model, next_state)) 
			
			Qs_update = Q_predict
			Qs_update[0][action] = new_Q
			model = TD_update(model, state, Qs_update)
			if done:
				score_episodes.append(score)
				break
			state = next_state
	print("\nTraining Completed")
	return model


env = gym.make("CartPole-v0")

num_train_episodes = 300000
test_episodes = 100

model = DeepQLearning(env, num_train_episodes)

model.save("MountainCar.h5")

