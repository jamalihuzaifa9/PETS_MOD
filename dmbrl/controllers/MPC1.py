from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import tensorflow as tf
import numpy as np
from scipy.io import savemat

from .Controller import Controller
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.optimizers import RandomOptimizer, CEMOptimizer


class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, params):
        """Creates class instance.

        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .update_fns (list<func>): A list of functions that will be invoked
                    (possibly with a tensorflow session) every time this controller is reset.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM, Random].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                .log_cfg
                    .save_all_models (bool): (optional) If True, saves models at every iteration.
                        Defaults to False (only most recent model is saved).
                        Warning: Can be very memory-intensive.
                    .log_traj_preds (bool): (optional) If True, saves the mean and variance of predicted
                        particle trajectories. Defaults to False.
                    .log_particles (bool) (optional) If True, saves all predicted particles trajectories.
                        Defaults to False. Note: Takes precedence over log_traj_preds.
                        Warning: Can be very memory-intensive
        """
        super().__init__(params)
        self.dO, self.dU = params.env.observation_space.shape[0], params.env.action_space.shape[0]
        self.ac_ub, self.ac_lb = params.env.action_space.high, params.env.action_space.low
        self.ac_ub = np.minimum(self.ac_ub, params.get("ac_ub", self.ac_ub))
        self.ac_lb = np.maximum(self.ac_lb, params.get("ac_lb", self.ac_lb))
        self.update_fns = params.get("update_fns", [])
        self.per = params.get("per", 1)

        self.model = get_required_argument(
            params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        )(params.prop_cfg.model_init_cfg)
        self.model_train_cfg = params.prop_cfg.get("model_train_cfg", {})
        self.prop_mode = get_required_argument(params.prop_cfg, "mode", "Must provide propagation method.")
        self.npart = get_required_argument(params.prop_cfg, "npart", "Must provide number of particles.")
        self.ign_var = params.prop_cfg.get("ign_var", False) or self.prop_mode == "E"

        self.obs_preproc = params.prop_cfg.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out)
        self.obs_postproc2 = params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs)
        self.targ_proc = params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs)

        self.opt_mode = get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.plan_hor = get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.obs_cost_fn = get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        self.ac_cost_fn = get_required_argument(params.opt_cfg, "ac_cost_fn", "Must provide cost on actions.")

        self.save_all_models = params.log_cfg.get("save_all_models", False)
        self.log_traj_preds = params.log_cfg.get("log_traj_preds", False)
        self.log_particles = params.log_cfg.get("log_particles", False)

        # Perform argument checks
        if self.prop_mode not in ["E", "DS", "MM", "TS1", "TSinf"]:
            raise ValueError("Invalid propagation method.")
        if self.prop_mode in ["TS1", "TSinf"] and self.npart % self.model.num_nets != 0:
            raise ValueError("Number of particles must be a multiple of the ensemble size.")
        if self.prop_mode == "E" and self.npart != 1:
            raise ValueError("Deterministic propagation methods only need one particle.")

        # Create action sequence optimizer
        # To see opt_cfg:
        #       cfg.ctrl_cfg.opt_cfg.get("cfg", {}) -> {'popsize': 200, 'num_elites': 40, 'max_iters': 5, 'alpha': 0.1}
        opt_cfg = params.opt_cfg.get("cfg", {})
        # params.opt_cfg.mode = CEM
        self.optimizer = MPC.optimizers[params.opt_cfg.mode](
            sol_dim=self.plan_hor*self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            tf_session=None if not self.model.is_tf_model else self.model.sess, **opt_cfg)
        self.GAMMA = tf.constant(1., dtype='float32')
        self.EPSILON_W = tf.constant(0.1, dtype='float32')
        self.net_number = 0

        # Controller state variables
        self.has_been_trained = params.prop_cfg.get("model_pretrained", False)
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])
        self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        )
        if self.model.is_tf_model:
            self.sy_cur_obs = tf.Variable(np.zeros(self.dO), dtype=tf.float32)
            self.ac_seq = tf.placeholder(shape=[1, self.plan_hor*self.dU], dtype=tf.float32)
            self.pred_cost, self.pred_traj = self._compile_cost(self.ac_seq, get_pred_trajs=True)
            self.optimizer.setup(self._compile_cost, True)
            self.model.sess.run(tf.variables_initializer([self.sy_cur_obs]))
        else:
            raise NotImplementedError()

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            rews_trajs: A list of reward arrays.

        Returns: None.
        """
        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
            new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

        # Train the model
        self.model.train(self.train_in, self.train_targs, **self.model_train_cfg)
        self.has_been_trained = True

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.optimizer.reset()
        if self.model.is_tf_model:
            for update_fn in self.update_fns:
                update_fn(self.model.sess)

    def act(self, obs, t, get_pred_cost=False):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        if not self.has_been_trained:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        if self.model.is_tf_model:
            self.sy_cur_obs.load(obs, self.model.sess)

        # Length = plan_horizon = 25
        soln, z = self.optimizer.obtain_solution(self.prev_sol, self.init_var)

        # Remove the first (self.per*self.dU) elements and add at the and a zero
        self.prev_sol = np.concatenate([np.copy(soln)[self.per*self.dU:], np.zeros(self.per*self.dU)])
        #self.prev_sol = tf.concat([soln[self.per * self.dU:], tf.zeros(self.per * self.dU)], axis=0)
        self.ac_buf = soln[:self.per*self.dU].reshape(-1, self.dU)

        if get_pred_cost and not (self.log_traj_preds or self.log_particles):
            if self.model.is_tf_model:
                pred_cost = self.model.sess.run(
                    self.pred_cost,
                    feed_dict={self.ac_seq: soln[None]}
                )[0]
            else:
                raise NotImplementedError()
            return self.act(obs, t), pred_cost
        elif self.log_traj_preds or self.log_particles:
            pred_cost, pred_traj = self.model.sess.run(
                [self.pred_cost, self.pred_traj],
                feed_dict={self.ac_seq: soln[None]}
            )
            pred_cost, pred_traj = pred_cost[0], pred_traj[:, 0]
            if self.log_particles:
                self.pred_particles.append(pred_traj)
            else:
                self.pred_means.append(np.mean(pred_traj, axis=1))
                self.pred_vars.append(np.mean(np.square(pred_traj - self.pred_means[-1]), axis=1))
            if get_pred_cost:
                return self.act(obs, t), pred_cost
        return self.act(obs, t)

    def dump_logs(self, primary_logdir, iter_logdir):
        """Saves logs to either a primary log directory or another iteration-specific directory.
        See __init__ documentation to see what is being logged.

        Arguments:
            primary_logdir (str): A directory path. This controller assumes that this directory
                does not change every iteration.
            iter_logdir (str): A directory path. This controller assumes that this directory
                changes every time dump_logs is called.

        Returns: None
        """
        self.model.save(iter_logdir if self.save_all_models else primary_logdir)
        if self.log_particles:
            savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
            self.pred_particles = []
        elif self.log_traj_preds:
            savemat(
                os.path.join(iter_logdir, "predictions.mat"),
                {"means": self.pred_means, "vars": self.pred_vars}
            )
            self.pred_means, self.pred_vars = [], []

    def _compile_cost(self, ac_seqs, get_pred_trajs=False):
        t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
        init_costs = tf.zeros([nopt, self.npart])
        returns = tf.zeros([nopt, self.npart])
        ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU])
        ac_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_seqs, [1, 0, 2])[:, :, None],
            [1, 1, self.npart, 1]
        ), [self.plan_hor, -1, self.dU])
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])
        
        ac_traj = tf.zeros(tf.shape(ac_seqs[0]))[None]

        #state_traj = tf.tile(init_obs[None], [25, 1000, 4])
        state_traj = init_obs[None]
        #state_traj = tf.TensorArray(dtype=tf.float32, size=self.optimizer.sol_dim)
        #current_cost_traj = tf.zeros([25, self.optimizer.popsize * self.npart, 1])
        current_cost_traj = tf.zeros([1, self.optimizer.popsize * self.npart, 1])
        #current_cost_traj = tf.TensorArray(dtype=tf.float32, size=self.optimizer.sol_dim, clear_after_read=False)
        #grads_traj = tf.zeros(tf.shape(self._gradients_reg(init_obs, ac_seqs[0])))

        def continue_prediction(t, *args):
            return tf.less(t, self.plan_hor)

        if get_pred_trajs:
            pred_trajs = init_obs[None]

            def iteration(t, total_cost, cur_obs, pred_trajs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                next_obs = self.obs_postproc2(next_obs)
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
                return t + 1, total_cost + (tf.math.pow(
                    self.GAMMA, tf.cast(t, dtype='float32')) * delta_cost), next_obs, pred_trajs
            _, costs, _, pred_trajs = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs, pred_trajs],
                shape_invariants=[
                    t.get_shape(), init_costs.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO])
                ]
            )

            # Replace nan costs with very high cost
            pred_trajs = tf.reshape(pred_trajs, [self.plan_hor + 1, -1, self.npart, self.dO])
            costs = tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs)
            costs4 = tf.reduce_mean(costs, axis=1)
            return costs4, pred_trajs
        else:
            '''
            def iteration(t, total_cost, cur_obs, state_traj, current_cost_traj):
                cur_acs = ac_seqs[t]
                #tf.concat([ac_traj, cur_acs[None]], axis = 0)
                #ac_traj.append(cur_acs)
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                state_traj = tf.concat([state_traj, next_obs[None]], axis=0)
                #state_traj[t] = next_obs[None]
                #state_traj.write(tf.cast(t, dtype='int32'), next_obs)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                #current_cost_traj = current_cost_traj.write(tf.cast(t, dtype='int32'), tf.reshape(delta_cost, [-1, self.optimizer.popsize * self.npart, 1]))
                current_cost_traj = tf.concat([current_cost_traj, tf.reshape(delta_cost, [-1, self.optimizer.popsize*self.npart, 1])], axis=0)
                #current_cost_traj[t] = tf.reshape(delta_cost, [-1, self.optimizer.popsize * self.npart, 1])
                b = tf.stack(tf.reshape(delta_cost, [-1, self.optimizer.popsize*self.npart, 1]))
                self.a[self.i] = b
                self.i = self.i +1
                # t + 1, total_cost + delta_cost, self.obs_postproc2(next_obs)
                return t + 1, total_cost + (tf.math.pow(
                    self.GAMMA, tf.cast(t, dtype='float32')) * delta_cost), self.obs_postproc2(next_obs), state_traj, current_cost_traj

            _, costs, _, state_traj, current_cost_traj = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs, state_traj, current_cost_traj],
                shape_invariants=[
                    t.get_shape(), init_costs.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO]),
                    tf.TensorShape([None, None, self.dU])
                ]
            )
            '''
            
            grad = []
            
            for t in range(self.plan_hor):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(init_obs, cur_acs)
                state_traj = tf.concat([state_traj, next_obs[None]], axis=0)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                c_traj = (tf.math.pow(self.GAMMA, tf.cast(t, dtype='float32')) * delta_cost)
                current_cost_traj = tf.concat(
                    [current_cost_traj, tf.reshape(c_traj, [-1, self.optimizer.popsize * self.npart, 1])], axis=0)
                init_costs = init_costs + (tf.math.pow(
                    self.GAMMA, tf.cast(t, dtype='float32')) * delta_cost)
                t = t + 1
                init_obs = self.obs_postproc2(next_obs)

            average_state_traj = tf.reduce_mean(state_traj, axis=0)

            for j in range(self.model.num_nets):
                currTensor = self._expand_to_ts_format(average_state_traj)[self.net_number]
                #     self.z = self.model.get_optvars(j)
                # grads = tf.map_fn(self._gradients_reg, currTensor)
                gradients=[]
                for i in range(currTensor.get_shape()[0]):
                    zz = tf.gradients(currTensor[i], self.model.optvars[:-2])
                    paramsGrad = [zz[k] for k in range(len(zz))]
                    paramGrad_norm = tf.linalg.global_norm(paramsGrad)
                    gradients.append(paramGrad_norm)
                self._update_net_number()
            
                grad.append(gradients)

            # print(tf.shape(currTensor))

            current_cost_traj = tf.where(tf.is_nan(current_cost_traj), 1e6 * tf.ones_like(current_cost_traj), current_cost_traj)
            costs = init_costs
            costs = tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs)


            reg = tf.zeros([self.model.num_nets, self.optimizer.popsize, 1])
            for i in range(self.plan_hor):
                rtg = tf.reduce_sum(current_cost_traj[i+1:self.plan_hor], axis=0)
                rtg = self._expand_to_ts_format(rtg)
                res = []
                for j in range(rtg.get_shape()[0]):
                    temp = tf.norm(rtg[j], axis = 1, keepdims=True)
                    temp2 = tf.reshape(grad[j], [-1,1])
                    temp3 = tf.math.multiply(temp, temp2)
                    res.append(temp3)
                res2 = tf.reshape(res, [self.model.num_nets, self.optimizer.popsize, -1, 1])
                res3 = tf.reduce_mean(res2, axis=2)
                reg = reg + res3

            #reg3 = tf.reduce_sum(reg, axis=0)
            reg4 = tf.reduce_mean(reg, axis=0)
            reg5 = tf.reduce_mean(reg4, axis=1)
            #reg5 = tf.reshape(reg4, [self.optimizer.popsize])


            costs4 = tf.reduce_mean(costs, axis=1) # average over ensemble
            obj = costs4 - (self.EPSILON_W*tf.math.sqrt(reg5))
            # obj = costs4
            regul = 0
            return obj, regul

    def _update_net_number(self):
        if self.net_number == self.model.num_nets-1:
            self.net_number = 0
        else:
            self.net_number = self.net_number + 1

    def _predict_next_obs(self, obs, acs):
        proc_obs = self.obs_preproc(obs)

        if self.model.is_tf_model:
            # TS Optimization: Expand so that particles are only passed through one of the networks.
            proc_obs, acs = self._expand_to_ts_format(proc_obs), self._expand_to_ts_format(acs)
            # Obtain model predictions
            inputs = tf.concat([proc_obs, acs], axis=-1)
            mean, var = self.model.create_prediction_tensors(inputs)
            if self.model.is_probabilistic and not self.ign_var:
                predictions = mean + tf.random_normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)
            else:
                predictions = mean

            # TS Optimization: Remove additional dimension
            predictions = self._flatten_to_matrix(predictions)
            return self.obs_postproc(obs, predictions)
        else:
            raise NotImplementedError()

    # def _get_optvars(self):
    #     z = [tf.identity(self.model.get_optvars()[i]) for i in range(len(self.model.optvars[:-2]))]
    #     return z

    def _expand_to_ts_format(self, mat):
        dim = mat.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(mat, [-1, self.model.num_nets, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [self.model.num_nets, -1, dim]
        )

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(ts_fmt_arr, [self.model.num_nets, -1, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [-1, dim]
        )

    def _prepare_gradients_reg(self, obs, acs):
        proc_obs = self.obs_preproc(obs)
        proc_obs, acs = self._expand_to_ts_format(proc_obs), self._expand_to_ts_format(acs)
        inputs = tf.concat([proc_obs, acs], axis=-1)
        mean, var = self.model.create_prediction_tensors(inputs)
        if self.model.is_probabilistic and not self.ign_var:
            predictions = mean + tf.random_normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)
        else:
            predictions = mean
        predictions = self._flatten_to_matrix(predictions)
        predictions2 = self.obs_postproc(obs, predictions)
        predictions3 = self._expand_to_ts_format(predictions2)
        return predictions3

    # def _gradients_reg(self, tensorState):
    #     zz = tf.gradients(tensorState, self.z)
    #     paramsGrad = [zz[k][self.net_number] for k in range(len(zz))]
    #     paramGrad_norm = tf.linalg.global_norm(paramsGrad)
    #     return paramGrad_norm


    
    
    def _gradients_reg(self, tensorState):
        gradients = []
        for i in range(tensorState.get_shape()[0]):
            zz = tf.gradients(tensorState[i], self.model.optvars[:-2])
            paramsGrad = [zz[k][i] for k in range(len(zz))]
            paramGrad_norm = tf.linalg.global_norm(paramsGrad)
            gradients.append(paramGrad_norm)
        return gradients
    
    # def _gradients_reg(self, tensorState):
    #     zz = tf.gradients(tensorState, self.model.optvars[:-2])
    #     # paramsGrad = [zz[k][i] for k in range(len(zz))]
    #     # paramGrad_norm = tf.linalg.global_norm(paramsGrad)
    #     # gradients.append(paramGrad_norm)
    #     return zz
        
    # def _gradients_reg(self, obs, acs):
    #     proc_obs = self.obs_preproc(obs)
    #     proc_obs, acs = self._expand_to_ts_format(proc_obs), self._expand_to_ts_format(acs)
    #     inputs = tf.concat([proc_obs, acs], axis=-1)
    #     mean, var = self.model.create_prediction_tensors(inputs)
    #     if self.model.is_probabilistic and not self.ign_var:
    #         predictions = mean + tf.random_normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)
    #     else:
    #         predictions = mean
    #     predictions = self._flatten_to_matrix(predictions)
    #     predictions2 = self.obs_postproc(obs, predictions)
    #     predictions3 = self._expand_to_ts_format(predictions2)
    #     gradients = []
    #     for i in range(predictions3.get_shape()[0]):
    #         gradients_nn = []
    #         for j in range(predictions3.get_shape()[1]):
    #             zz = tf.gradients(predictions3[i][j], self.model.optvars[:-2])
    #             paramsGrad = [zz[k][i] for k in range(len(zz))]
    #             paramGrad_norm = tf.linalg.global_norm(paramsGrad)
    #             gradients_nn.append(paramGrad_norm)
    #         gradients.append(gradients_nn)
    #     return gradients

    # def _compile_cost(self, ac_seqs, get_pred_trajs=False):
    #     t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
    #     init_costs = tf.zeros([nopt, self.npart])
    #     ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU])
    #     ac_seqs = tf.reshape(tf.tile(
    #         tf.transpose(ac_seqs, [1, 0, 2])[:, :, None],
    #         [1, 1, self.npart, 1]
    #     ), [self.plan_hor, -1, self.dU])
    #     init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])

    #     def continue_prediction(t, *args):
    #         return tf.less(t, self.plan_hor)

    #     if get_pred_trajs:
    #         pred_trajs = init_obs[None]

    #         def iteration(t, total_cost, cur_obs, pred_trajs):
    #             cur_acs = ac_seqs[t]
    #             next_obs = self._predict_next_obs(cur_obs, cur_acs)
    #             delta_cost = tf.reshape(
    #                 self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
    #             )
    #             next_obs = self.obs_postproc2(next_obs)
    #             pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
    #             return t + 1, total_cost + (tf.math.pow(
    #                 self.GAMMA, tf.cast(t, dtype='float32')) * delta_cost), next_obs, pred_trajs

    #         _, costs, _, pred_trajs = tf.while_loop(
    #             cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs, pred_trajs],
    #             shape_invariants=[
    #                 t.get_shape(), init_costs.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO])
    #             ]
    #         )

            # Replace nan costs with very high cost
        #     costs = tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)
        #     pred_trajs = tf.reshape(pred_trajs, [self.plan_hor + 1, -1, self.npart, self.dO])
        #     return costs, pred_trajs
        # else:
        #     def iteration(t, total_cost, cur_obs):
        #         cur_acs = ac_seqs[t]
        #         next_obs = self._predict_next_obs(cur_obs, cur_acs)
        #         delta_cost = tf.reshape(
        #             self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
        #         )
        #         return t + 1, total_cost + (tf.math.pow(
        #             self.GAMMA, tf.cast(t, dtype='float32')) * delta_cost), self.obs_postproc2(next_obs)

        #     _, costs, _ = tf.while_loop(
        #         cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs]
        #     )

        #     # Replace nan costs with very high cost
        #     costs = tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)
        #     return costs