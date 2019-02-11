'''https://github.com/fmfn/BayesianOptimization'''
# Support for maths
import numpy as np
# Plotting tools
from matplotlib import pyplot as plt
# we use the following for plotting figures in jupyter
# GPy: Gaussian processes library
import GPy

from bayes_opt import BayesianOptimization
from mnist import MultinomialLogisticRegressionClassifier

mnist_classifier=MultinomialLogisticRegressionClassifier()

# Bounded region of parameter space
''' pbounds variables need to have same names as black_box_function's attributes'''
pbounds = {'x1': (-2, -5), 'x2': (-1, -9)}



def black_box_function(x1, x2):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    # hyper-parameters, learning rate and l1 penalty coefficient
    # transform from log range to normal range
    x1=10**x1
    x2=10**x2
    return mnist_classifier.nn_train(x1, x2)


optimizer = BayesianOptimization(
f=black_box_function,
pbounds=pbounds,
verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
random_state=1,
)


optimizer.maximize(
    init_points=2,
    n_iter=3,
)

print(optimizer.max)