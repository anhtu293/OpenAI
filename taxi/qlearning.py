import gym
import numpy as np
import sys
import time
import itertools
from gym.envs.toy_text.taxi import TaxiEnv
from matplotlib import pyplot as plt
from collections import defaultdict

def nth_root(num, n):
	return(n**(1/num))

#epsilon greedy policy
def epsilon_greedy_policy(Q, epsilon, nA, state):
	A = np.ones(nA, dtype = float) * epsilon / nA
	best_action = np.argmax(Q[state])
	A[best_action] += (1 - epsilon)
	return A

def q_learning(env, num_episodes, discount_factor = 1.0, alpha = 0.5, epsilon = 1.0):
	Q = defaultdict(lambda : np.zeros(env.action_space.n))
	print("Start training \n")
	time.sleep(0.5)
	final_epsilon = 0.01
	decay_epsilon = nth_root(num_episodes, final_epsilon/epsilon)
	episodes_reward = np.zeros((num_episodes,1))
	for i_episode in range(num_episodes):
		if (i_episode + 1) % 10 == 0:
			print("\r Episode {}/{}".format(i_episode + 1, num_episodes), end = "")
			sys.stdout.flush()

		state = env.reset()
		for t in itertools.count():
			epsilon = epsilon * decay_epsilon
			action_probs = epsilon_greedy_policy(Q, epsilon, env.action_space.n, state)
			action = np.random.choice(range(len(action_probs)), p = action_probs)
			next_state, reward, done, _ = env.step(action)

			episodes_reward[i_episode] += reward

			best_action_next = np.argmax(Q[next_state])
			update_term = reward + discount_factor*Q[next_state][best_action_next] - Q[state][action]

			Q[state][action] += alpha*update_term

			if done:
				break
			state = next_state

	return Q, episodes_reward

def test_policy(Q, env, num_episodes):
	success = 0
	print("Start test \n")
	for i_episode in range(num_episodes):
		state = env.reset()
		for t in itertools.count():
			action = np.argmax(Q[state])
			next_state, reward, done, _ = env.step(action)
			if done:
				if reward > 0:
					success += 1
				break
			state = next_state
	print("\nNumber of episodes : {} \n Success : {} \n Successful rate : {}".format(num_episodes, success, success*100/num_episodes))

env = gym.make("Taxi-v2")
num_episodes_train = 200000
num_episodes_test = 2000
Q, episodes_reward = q_learning(env, num_episodes_train)

test_policy(Q, env, num_episodes_test)