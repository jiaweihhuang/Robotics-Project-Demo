#!/usr/bin/env python3

# from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import RLBlocks.tf_util as tf_util as U
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
import gym


def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env


def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import RLBlocks.mlp_policy as mlp_policy , pposgd_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=2048,
                        clip_param=0.2, entcoeff=0.0,
                        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                        gamma=0.99, lam=0.95, schedule='linear',
                        )
    env.close()


def main():
    # args = mujoco_arg_parser().parse_args()
    env = 'HalfCheetah-v1'
    num_timesteps = 1e6
    seed = 0
    logger.configure()
    train(env, num_timesteps=num_timesteps, seed=seed)


if __name__ == '__main__':
    main()
