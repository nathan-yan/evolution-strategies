import numpy as np 
import time
import gym 

from gym import wrappers

generations = 1000
population_size = 100
observation_size = 4

def relu(x):
    return (x > 0) * x

def model(inp, weights):
    fc1 = relu(np.dot(inp, weights[0]))
    fc2 = np.dot(fc1, weights[1])

    return fc2

weights_1 = np.load('baseweight_0m.npy')
weights_2 = np.load('baseweight_1m.npy')

weights = [weights_1, weights_2]

def main():
    env = gym.make('MountainCar-v0')
    #env = wrappers.Monitor(env, 'acrobot-1')

    for generation in range(generations):
        observation = env.reset()
        r = 0
        
        while True:
        #for i in range (150):
            env.render()
            action = np.argmax(model(observation, weights))
            observation, reward, done, info = env.step(action)
            r += reward
            if done: print r; break;
            #print reward
    
   # gym.upload('acrobot-1', api_key='sk_dI2ter1rS2qjRoovttM4w')

main()