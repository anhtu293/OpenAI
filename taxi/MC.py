import gym
import numpy as np
from collections import defaultdict
import time
import sys
import itertools
from gym.envs.toy_text.taxi import TaxiEnv

def nth_root(num, n):
	return(n**(1/num))

def epsilon_greedy_policy(Q, epsilon, nA, state):
	A = np.ones(nA, dtype = float) * epsilon / nA
	best_action = np.argmax(Q[state])
	A[best_action] += 1 - epsilon
	return A

def mc_control_epsilon_greedy(env, num_episodes, discount_factor = 1.0, epsilon = 1):
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)
	final_epsilon = 0.01
	epsilon_decay = nth_root(num_episodes, final_epsilon/epsilon)

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	print("\r Start Training")
	time.sleep(1)
	for i_episode in range(num_episodes):
		if (i_episode + 1) % 100 == 0:
			print("\r Episode {}/{}".format(i_episode + 1, num_episodes), end = "")
			sys.stdout.flush()

		#Decay epsilon
		epsilon = epsilon * epsilon_decay
		#create episode
		episode = []
		state = env.reset()
		#policy = epsilon_geedy_policy(Q, epsilon, nA, state)
		action_probs = epsilon_greedy_policy(Q, epsilon, env.action_space.n, state)
		action = np.random.choice(range(len(action_probs)), p = action_probs)
		next_state, reward, done, _ = env.step(action)
		episode.append((next_state, action, reward))
		state = next_state
		while not done:
			action_probs = epsilon_greedy_policy(Q, epsilon, env.action_space.n, state)
			action = np.random.choice(range(len(action_probs)), p = action_probs)
			next_state, reward, done, _ = env.step(action)
			episode.append((state, action, reward))
			state = next_state

		state_action_set = set([(x[0], x[1]) for x in episode])
		for state, action in state_action_set:
			sa_pair = (state, action)
			index_first_visit = next(i for i,x in enumerate(episode) if x[0] == state)
			G = sum([x[2] * discount_factor**i for i,x in enumerate(episode[index_first_visit:])])
			returns_sum[sa_pair] += G
			returns_count[sa_pair] += 1
			Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
	return Q

def test_policy(env, Q, num_episodes):
	success = 0
	print("\n Start Test")
	for i_episode in range(num_episodes):
		state = env.reset()
		print("\r Episode {}/{}".format(i_episode, num_episodes), end = "\n")
		for t in itertools.count():
			action = np.argmax(Q[state])
			next_state, reward, done, _ = env.step(action)
			#env.render()
			#time.sleep(0.3)
			if done:
				if reward > 0:
					success += 1
				break
			state = next_state
	print("\nNumber of episodes : {} \n Success : {} \n Successful rate : {}".format(num_episodes, success, success*100/num_episodes))


env = gym.make("Taxi-v2")
train_num_episodes = 300000
test_num_episodes = 2000

Q = mc_control_epsilon_greedy(env, train_num_episodes)
#print(Q)

test_policy(env, Q, test_num_episodes)

