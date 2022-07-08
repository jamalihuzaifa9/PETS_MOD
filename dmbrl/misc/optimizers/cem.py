from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.stats as stats

from .optimizer import Optimizer


class CEMOptimizer(Optimizer):
    """A Tensorflow-compatible CEM optimizer.
    """
    def __init__(self, sol_dim, max_iters, popsize, num_elites, tf_session=None,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.tf_sess = tf_session
        self.z = []

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver") as scope:
                    self.init_mean = tf.placeholder(dtype=tf.float32, shape=[sol_dim])
                    self.init_var = tf.placeholder(dtype=tf.float32, shape=[sol_dim])

        self.num_opt_iters, self.mean, self.var = None, None, None
        self.tf_compatible, self.cost_function = None, None

    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        if tf_compatible and self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        self.tf_compatible = tf_compatible

        self.cost_function = cost_function # DELETE THIS
        if not tf_compatible:
            self.cost_function = cost_function
        else:
            def continue_optimization(t, mean, var, best_val, best_sol):
                return tf.logical_and(tf.less(t, self.max_iters), tf.reduce_max(var) > self.epsilon)

            # @tf.function
            def iteration(iter, mean, var, best_val, best_sol):
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
                samples = tf.truncated_normal([self.popsize, self.sol_dim], mean, tf.sqrt(constrained_var))

                costs, self.z = cost_function(samples)
                values, indices = tf.nn.top_k(-costs, k=self.num_elites, sorted=True)

                best_val, best_sol = tf.cond(
                    tf.less(-values[0], best_val),
                    lambda: (-values[0], samples[indices[0]]),
                    lambda: (best_val, best_sol)
                )

                elites = tf.gather(samples, indices)
                new_mean = tf.reduce_mean(elites, axis=0)
                new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                return iter + 1, mean, var, best_val, best_sol

            with self.tf_sess.graph.as_default():
                self.num_opt_iters, self.mean, self.var, self.best_val, self.best_sol = tf.while_loop(
                    cond=continue_optimization, body=iteration,
                    loop_vars=[0, self.init_mean, self.init_var, float("inf"), self.init_mean]
                )

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if self.tf_compatible:
            sol, solvar = self.tf_sess.run(
                [self.mean, self.var],
                feed_dict={self.init_mean: init_mean, self.init_var: init_var}
            )
        elif self.tf_compatible and 0:
            def proceed(mean0, var0):
                    lb_dist, ub_dist = mean0 - self.lb, self.ub - mean0
                    constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var0)

                    samples = tf.truncated_normal([self.popsize, self.sol_dim], mean0, tf.sqrt(constrained_var))
                    costs, self.z = self.cost_function(samples)
                    values, indices = tf.nn.top_k(-costs, k=self.num_elites, sorted=True)

                    # best_val, best_sol = tf.cond(
                    #    tf.less(-values[0], best_val),
                    #    lambda: (-values[0], samples[indices[0]]),
                    #    lambda: (best_val, best_sol)
                    # )

                    elites = tf.gather(samples, indices)
                    new_mean = tf.reduce_mean(elites, axis=0)
                    new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)

                    mean = self.alpha * mean0 + (1 - self.alpha) * new_mean
                    var = self.alpha * var0 + (1 - self.alpha) * new_var

                    return mean, var

            mean, var, t = self.init_mean, self.init_var, 0
            # while (t < self.max_iters):
            while (t < 1):
                print(t)
                mean, var = tf.cond(
                    tf.reduce_max(var) > self.epsilon,
                    lambda: proceed(mean, var),
                    lambda: (mean, var)
                )
                t += 1
            sol, solvar = mean, var
        elif self.tf_compatible and 0:
            def continue_optimization(iter, mean, var, best_val, best_sol):
                return tf.logical_and(tf.less(iter, self.max_iters), tf.reduce_max(var) > self.epsilon)

            def iteration(iter, mean, var, best_val, best_sol):
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
                samples = tf.truncated_normal([self.popsize, self.sol_dim], mean, tf.sqrt(constrained_var))

                costs, self.z = self.cost_function(samples)
                values, indices = tf.nn.top_k(-costs, k=self.num_elites, sorted=True)

                best_val, best_sol = tf.cond(
                    tf.less(-values[0], best_val),
                    lambda: (-values[0], samples[indices[0]]),
                    lambda: (best_val, best_sol)
                )

                elites = tf.gather(samples, indices)
                new_mean = tf.reduce_mean(elites, axis=0)
                new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                return iter + 1, mean, var, best_val, best_sol

            with self.tf_sess.graph.as_default():
                self.num_opt_iters, self.mean, self.var, self.best_val, self.best_sol = tf.while_loop(
                    cond=continue_optimization, body=iteration,
                    loop_vars=[0, self.init_mean, self.init_var, float("inf"), self.init_mean]
                )
            sol, solvar = self.mean, self.var
        elif self.tf_compatible and 0:
            mean, var, t = init_mean, init_var, 0
            zz = tf.logical_and(tf.less(t, self.max_iters), tf.reduce_max(var) > self.epsilon)
            with self.tf_sess.as_default():
                z = zz.eval()
            while z:
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
                samples = tf.truncated_normal([self.popsize, self.sol_dim], mean, tf.cast(constrained_var, dtype='float32'))

                costs, self.z = self.cost_function(samples)
                values, indices = tf.nn.top_k(-costs, k=self.num_elites, sorted=True)

                #best_val, best_sol = tf.cond(
                #    tf.less(-values[0], best_val),
                #    lambda: (-values[0], samples[indices[0]]),
                #    lambda: (best_val, best_sol)
                #)

                elites = tf.gather(samples, indices)
                new_mean = tf.reduce_mean(elites, axis=0)
                new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                t += 1
            sol, solvar = mean, var

        elif self.tf_compatible and 0:
            sol, solvar = self.tf_sess.run(
                [self.mean, self.var],
                feed_dict={self.init_mean: init_mean, self.init_var: init_var}
            )
        else:
            mean, var, t = init_mean, init_var, 0
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

            while (t < self.max_iters) and np.max(var) > self.epsilon:
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

                samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
                costs, self.z = self.cost_function(samples)
                elites = samples[np.argsort(costs)][:self.num_elites]

                new_mean = np.mean(elites, axis=0)
                new_var = np.var(elites, axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                t += 1
            sol, solvar = mean, var
        return sol, self.z
