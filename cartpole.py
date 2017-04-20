import numpy as np 

import gym 

generations = 1000
population_size = 100
observation_size = 4

def relu(x):
    return (x > 0) * x

def model(inp, weights):
    fc1 = relu(np.dot(inp, weights[0]))
    fc2 = np.dot(fc1, weights[1])

    return fc2

def main():
    env = gym.make('CartPole-v0')

    base_weight = [0.01 * np.random.randn(4, 10), 0.01 * np.random.randn(10, 2)]

    gaussian = [[np.random.randn(4, 10) * 0.2, np.random.randn(10, 2) * 0.2] for a in range (population_size)]
    perturbed_weights = [[gaussian[a][0] + base_weight[0], gaussian[a][1] + base_weight[1]] for a in range (population_size)]

    for generation in range(generations):
        accumulated_rewards = [] 
        average_r = 0 
        for agent in range (population_size):
            w = perturbed_weights[agent]
            r = 0

            observation = env.reset()
            for t in range (200):
                if generation % 100 == 0 and agent % 10 == 0:
                    env.render() 

                action = np.argmax(model(observation, w))

                observation, reward, done, info = env.step(action)
                r += reward / 10.0

                if done:
                    r -= 1
                    break;
            
            #if agent % 5 == 0:
            #    print "Completed run for agent -- " + str(agent) + ' ( generation ' + str(generation) + ' )' + str(r)
            
            accumulated_rewards.append(r)
            average_r += r 
        
        print "Average reward -- " + str(average_r/population_size)

        average_1 = 0
        average_2 = 0
        
        average_accumulated_rewards = 0

        for agent in range (population_size):
            average_1 += accumulated_rewards[agent] * gaussian[agent][0]
            average_2 += accumulated_rewards[agent] * gaussian[agent][1]
        
        average_1 /= population_size
        average_2 /= population_size 

        base_weight[0] = base_weight[0] + 0.02 * average_1 * (generations - generation)/generations 
        base_weight[1] = base_weight[1] + 0.02 * average_2 * (generations - generation)/generations 

        np.save('baseweight_0c.npy', base_weight[0])
        np.save('baseweight_1c.npy', base_weight[1])

        gaussian = [[np.random.randn(4, 10) * 0.03, np.random.randn(10, 2) * 0.03] for a in range (population_size)]
        perturbed_weights = [[gaussian[a][0] + base_weight[0], gaussian[a][1] + base_weight[1]] for a in range (population_size)]

main()
