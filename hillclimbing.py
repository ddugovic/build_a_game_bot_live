import gym
import numpy as np

def run_episode(env, parameters):
	observation = env.reset()
	totalreward = 0
	#for 200 timesteps
	for _ in xrange(200):
		#env.render()
		#initalize random weights
		action = 0 if np.matmul(parameters, observation)<0 else 1
		observation, reward, done, info = env.step(action)
		totalreward += reward
		if done:
			break
	return totalreward

#hill climbing algo training
def train(submit):
	np.random.seed(0)
	outdir = '/tmp/gym/hill-climbing-agent-results'
	env = gym.make('CartPole-v0')
	env.monitor.start(outdir, force=True, seed=0)

	episodes_per_update = 5
	noise_scaling = 0.1
	parameters = np.zeros(env.observation_space.shape[0])
	bestreward = 0

	#1000 episodes
	for _ in xrange(1000):
		newparams = parameters + (np.random.rand(len(parameters)) * 2 - 1) * noise_scaling
		reward = run_episode(env, newparams)
		print "reward %d best %d" % (reward, bestreward)
		if reward > bestreward: 
			bestreward = reward
			parameters = newparams
			if reward >= 195.0:
				noise_scaling /= 2.0

	# Dump result info to disk
	env.monitor.close()

	# Upload to the scoreboard
	if submit:
		gym.upload(outdir)

r = train(submit=True)
print r
