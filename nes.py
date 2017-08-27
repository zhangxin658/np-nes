''' Natural Evolution Strategy (NES) '''

import sys
import argparse
import numpy as np

# Number of parameters (for this module's fitness function).
# This variable is only used when it is not given from outside fitness evaluation module.
n_param = 3


def fitness_function(param):
    ''' Example fitness evaluation function.

    Args:
        param: An np.ndarray that represents a list of parameters.
    Returns:
        A fitness value computed with a negative sum of squared error from target parameters.
    '''

    return -np.sum((np.array([0.5, 0.1, -0.3]) - param) ** 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness', type=str, help='Fitness evaluation module')
    parser.add_argument('--n_iter', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--lambda', dest='lambda_', type=int, default=50, help='Population size')
    parser.add_argument('--alpha', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sigma', type=float, default=0.1, help='Noise standard deviation')

    args = parser.parse_args()

    if not args.fit_func is None:
        try:
            fitness_module = __import__(args.fitness)
        except ImportError:
            print('Import %s failed: running demo...' % args.fitness)
        else:
            if not hasattr(fitness_module, 'fitness_function'):
                raise AttributeError('Fitness module must include \'fitness_function(param)\'.')
            if not hasattr(fitness_module, 'n_param'):
                raise AttributeError('Fitness module must include global variable \'n_param\'.')

            fitness_function = fitness_module.fitness_function
            n_param = fitness_module.n_param

    # random seed
    np.random.seed(0)

    # initial parameters
    param = np.random.randn(n_param)
    for i in range(args.n_iter):
        # sample population from the search distribution
        var = np.random.randn(args.lambda_, n_param)
        pop = np.array([param + args.sigma * var[i] for i in range(args.lambda_)])

        # evaluate each fitness value
        fitness = np.array([fitness_function(ind) for ind in pop])
        fit_normal = (fitness - np.mean(fitness)) / np.std(fitness)

        # gradient ascent
        param = param + args.alpha / (args.lambda_ * args.sigma) * np.dot(var.T, fit_normal)

        if i % (args.n_iter / 10) == 0:
            print('iter %i: fitness = %f' % (i, fitness_function(param)))
