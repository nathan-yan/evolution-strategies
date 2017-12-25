import numpy as np
import gym

import sys

def init_weights(i, o, sigma, mu):
    return np.random.randn(i, o) * sigma + mu

class Worker:
    def __init__(self, variance = 1, mean = 0):
        self.variance = variance
        self.mean = mean

        self.weights = {'fc1' : init_weights(24, 40, self.variance, self.mean),
                        'fc2' : init_weights(40, 40, self.variance, self.mean),
                        'fc3' : init_weights(40, 4, self.variance, self.mean)}

        self.acc_reward = 0
        self.reward = -1000000
        self.trials = 0

    def forward(self, inp):
        fc1 = np.dot(inp, self.weights['fc1'])
        fc1_activation = np.tanh(fc1)

        fc2 = np.dot(fc1_activation, self.weights['fc2'])
        fc2_activation = np.tanh(fc2)

        fc3 = np.tanh(np.dot(fc2_activation, self.weights['fc3']))

        return fc3

    def rollout(self, env, same_limit = 100, same_penalty = 100, ts_limit = 1000, ts_penalty = 100, render = False):
        acc_reward = 0
        done = False

        obv = env.reset()
        prev_obv = np.zeros_like(obv)

        ts = 0
        same = 0
        while not done:
            action = self.forward(obv)
            #action = np.argmax(action)

            obv, reward, done, info = env.step(action)

            if (np.sum((obv - prev_obv) ** 2) < 0.001):
                same += 1
                if same > same_limit:
                    acc_reward -= same_penalty
                    done = True
            else:
                same = 0

            if render:
                env.render()

            acc_reward += reward

            ts += 1
            if ts > ts_limit:
                acc_reward -= ts_penalty
                done = True

            prev_obv = obv

        self.acc_reward = acc_reward
        self.reward = acc_reward

        return acc_reward

    def reproduce(self, mutation_variance = 1, mutation_mean = 0):
        # reproduction is just mutation
        new_weights = {}
        for k in self.weights:
            delta = np.random.randn(*self.weights[k].shape) * mutation_variance + mutation_mean

            new_weights[k] = self.weights[k] + delta

        # produce a new worker
        worker = Worker()
        worker.weights = new_weights

        return worker

    def copy(self):
        return self.reproduce(mutation_variance = 0)

    # We've flipped these because the sort function sorts from smallest to greatest, which is the opposite of what we want: we want the workers with highest fitness to be at the top
    def __lt__(self, other):
        return other.reward < self.reward

    def __gt__(self, other):
        return not self.__lt__(other)

def main():
    generations = 1000     # there will be `generations` generations
    workers = 1000       # there are only `workers` workers in the population
    selection = 20      # top `selection` workers are eligible for selection for the next generation
    diversity = 10      # `diversity` random workers will be added to the population every generation

    population = [Worker() for w in range (workers)]

    env = gym.make("BipedalWalker-v2")

    for generation in range (generations):
        avg_reward = 0
        for worker in range (workers):
            w = population[worker]

            for t in range (1):
                reward = w.rollout(env, ts_limit = 100000, ts_penalty = 0)

            #print(reward)
            print("\r{0:{width}}".format("Evaluating worker " + str(worker) + " / " + str(workers), width = 10, fill = ' ', align = 'right'), end='', file=sys.stdout)
            #print("Evaluating worker " + str(worker) + " / " + str(workers),)
            sys.stdout.flush()  # flush is needed.

            avg_reward += w.reward

        print("")
        print("Evaluated generation " + str(generation + 1) + " / " + str(generations) + ". Average reward was " + str(avg_reward / workers))

        # after all workers have finished, perform selection
        # sort population by reward/fitness

        population.sort()

        # only the top `selection` workers are eligible for reproduction
        pool = population[:selection]

        print("Top " + str(selection) + " worker rewards")
        c = 0
        for p in pool:
            c += 1
            print(str(c) + " : " + str(np.round(p.reward, 3)), end = '   ')

        print("")


        print("Evaluating top worker")
        for i in range (5):
            print(str(i) + " : " + str(pool[0].rollout(env, ts_limit = 100000, ts_penalty = 0, render = True)), end = '   ')

        print("")
        print("")

        # clear population
        population = []

        for n in range (workers - (1 + diversity)):
            # randomly select a worker in the reproduction pool and reproduce it
            repro = np.random.choice(pool)
            pseudo_child = repro.reproduce(mutation_variance = 0.002)

            population.append(pseudo_child)

        for d in range (diversity):
            population.append(Worker())

        # elitism
        population.append(pool[0].copy())

main()
