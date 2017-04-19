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
    env = gym.make('Pendulum-v0')

    base_weight = [0.1 * np.random.randn(3, 10), 0.1 * np.random.randn(10, 1)]

    gaussian = [[np.random.randn(3, 10) * 0.1, np.random.randn(10, 1) * 0.1] for a in range (population_size)]
    perturbed_weights = [[gaussian[a][0] + base_weight[0], gaussian[a][1] + base_weight[1]] for a in range (population_size)]

    for generation in range(generations):
        accumulated_rewards = [] 
        average_r = 0 
        for agent in range (population_size):
            w = perturbed_weights[agent]
            r = 0

            observation = env.reset()
            for t in range (200):
                if generation % 100 == 0 and agent % 33 == 0:
                    env.render() 

                action = model(observation, w)

                observation, reward, done, info = env.step(action)
                if reward > 0:
                    r += reward * 2
                else:
                    r += reward

                if done:
                    break;
            
            #if agent % 5 == 0:
            #    print "Completed run for agent -- " + str(agent) + ' ( generation ' + str(generation) + ' )' + str(r)
            
            #r -= np.sum(base_weight[0] ** 2) + np.sum(base_weight[1] ** 2)

            accumulated_rewards.append(r)
            average_r += r 
            #print r
        
        print "Average reward -- " + str(sum(accumulated_rewards)/population_size) + str(max(accumulated_rewards))

        average_1 = 0
        average_2 = 0
        
        average_accumulated_rewards = average_r/population_size

        for agent in range (population_size):
            average_1 += (accumulated_rewards[agent] - average_accumulated_rewards) * gaussian[agent][0]
            average_2 += (accumulated_rewards[agent] - average_accumulated_rewards) * gaussian[agent][1]
        
        average_1 /= population_size
        average_2 /= population_size 

        base_weight[0] = base_weight[0] + 0.05 * average_1
        base_weight[1] = base_weight[1] + 0.05 * average_2
        np.save('baseweight_0.npy', base_weight[0])
        np.save('baseweight_1.npy', base_weight[1])
       # print base_weight

        gaussian = [[np.random.randn(3, 10) * 0.1, np.random.randn(10, 1) * 0.1] for a in range (population_size)]

        perturbed_weights = []
        for a in range (population_size):
            perturbed_weights.append([gaussian[a][0] + base_weight[0], gaussian[a][1] + base_weight[1]])
           # perturbed_weights.append([base_weight[0] - gaussian[a][0], base_weight[1] - gaussian[a][1]])

main()