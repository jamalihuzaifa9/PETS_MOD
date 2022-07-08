from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config

# %%
env = "cartpole"
ctrl_type = "MPC"
ctrl_args = []
overrides = []
logdir = 'log'

# %%
ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
cfg.ctrl_cfg.prop_cfg.model_init_cfg['num_nets'] = 5
cfg.ctrl_cfg['opt_cfg'].cfg['popsize'] = 10
# cfg.ctrl_cfg.prop_cfg.npart= 10
cfg.ctrl_cfg.prop_cfg.npart= 5
# cfg.ctrl_cfg['prop_cfg'].model_init_cfg['num_nets'] = 5
cfg.ctrl_cfg['prop_cfg'].model_init_cfg['num_nets'] = 5
cfg.exp_cfg.sim_cfg.task_hor = 100
cfg.pprint()

# %%
'''
cfg.ctrl_cfg.opt_cfg.plan_hor = 10
cfg.ctrl_cfg.opt_cfg['plan_hor'] = 10
cfg.ctrl_cfg.prop_cfg.model_train_cfg['epochs'] = 1
cfg.ctrl_cfg['prop_cfg'].model_train_cfg['epochs'] = 1
cfg.ctrl_cfg.prop_cfg.npart= 5

cfg.exp_cfg.exp_cfg.ntrain_iters = 10

cfg.exp_cfg.sim_cfg.shape.task_hor = 50
cfg.exp_cfg.sim_cfg.task_hor = 50
'''
# %%
if ctrl_type == "MPC":
    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
exp = MBExperiment(cfg.exp_cfg)

os.makedirs(exp.logdir)
with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
    f.write(pprint.pformat(cfg.toDict()))

# %%
exp.nrecord = 0
exp.run_experiment()