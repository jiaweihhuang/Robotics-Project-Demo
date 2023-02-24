import gym
import RLBlocks.logz as logz
import os
import tensorflow as tf
import baselines.common.tf_util as U
import numpy as np
import time
import argparse
import cloudpickle, tempfile, zipfile
import pybullet_data
import ray

from baselines.common import Dataset, explained_variance, zipsame
from collections import deque
from baselines.deepq.utils import load_state, save_state

from baselines.bench import Monitor
from baselines.common import set_global_seeds
from RLBlocks.normal_ppo import TF_Model
from RLBlocks.util import ActWrapper
import RLBlocks.tf_util as tf_util
from WorkerClass import Worker
from functools import partial
from copy import deepcopy

from gym.envs.registration import register
'''
python leopard_DeepMimic/trainSegments.py --random-scale 5.0 --noise-scale 0.0 --seg-index 0
'''

import gym_ext_envs.gym_dm

gym_register_func = partial(
    gym.envs.register,
    id='dm_leopard-v1',
    entry_point='gym_ext_envs.gym_dm.envs:DMLeopardEnv',
    max_episode_steps=10000,)
gym_register_func()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Overlap = 'Mid'
seg_start_frames = [0, 115, 260, 400, 550, 700,
                        850, 1090, 1250, 1460,
                        1650, 1780, 1950, 2120, 2300,
                        2450, 2750, 2900, 3100]

if Overlap == 'Mid':        
    seg_end_frames = [195, 361, 498, 655, 805, 955, 
                            1258, 1362, 1607, 1783, 1871, 
                            2069, 2239, 2426, 2555, 2960, 
                            3005, 3240, 3176]
elif Overlap == 'Short':                                
    seg_end_frames = [149, 303, 442, 595, 745, 895, 
                            1162, 1298, 1523, 1707, 1819, 
                            2001, 2171, 2354, 2495, 2840, 
                            2945, 3160, 3158]
elif Overlap == 'Long':   
    seg_end_frames = seg_start_frames[2:] + [1e6]                             
else:
    raise NotImplementedError

env_name = 'dm_leopard-v1'

def time_info(prt_info, last_time):
    print(prt_info, time.time() - last_time)
    return time.time()

def get_session():
    return tf.get_default_session()

def load_model(path):
    with open(path, "rb") as f:
        model_data = cloudpickle.load(f)
    with tempfile.TemporaryDirectory() as td:
        arc_path = os.path.join(td, "packed.zip")
        with open(arc_path, "wb") as f:
            f.write(model_data)

        zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
        load_state(os.path.join(td, "model"))

def make_mujoco_env(env_id, seed, logdir, args):
    motion_file = os.path.join(pybullet_data.getDataPath(), 'data', 'motions', 'leopard_retarget_motion.txt')
    
    set_global_seeds(seed)

    args.startFrame = seg_start_frames[args.seg_index]
    args.endFrame = seg_end_frames[args.seg_index]

    env = gym.make(env_id, motion_file=motion_file, periodic=False, tm_args=args)
    
    env.seed(seed)
    
    return env



def learn(env, timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          # annealing for stepsize parameters (epsilon and adam)
          schedule='linear', logdir=None,
          args=None, prefix=None,
          ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    env_func = partial(make_mujoco_env, env_id=env_name, logdir=args.logdir)
    
    print('the shape of ac_space is ', ac_space.shape)
    model = TF_Model(ob_space, ac_space, clip_param, entcoeff, args=args,
                    hid_size=args.hid_size, activation=args.activation,
                    optim_stepsize=optim_stepsize)

    U.initialize()

    worker_list = []
    eval_worker_list = []
    for i in range(args.num_workers):
        worker = Worker.remote(timesteps_per_actorbatch // args.num_workers, args=args,
                            hid_size=args.hid_size, num_hid_layers=2, gym_register_func=gym_register_func,
                            index=i, activation=args.activation, env_func=env_func)
        worker_list.append(worker)

        
    for i in range(args.num_workers):
        eval_args = deepcopy(args)
        # if args.random_scale == 1.0:
        #     eval_args.random_scale = 5.0
        # else:
        eval_args.random_scale = 1.0
        worker = Worker.remote(timesteps_per_actorbatch // args.num_workers, args=eval_args,
                            hid_size=args.hid_size, num_hid_layers=2, gym_register_func=gym_register_func,
                            index=i, activation=args.activation, env_func=env_func)
        eval_worker_list.append(worker)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards
    eval_lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    eval_rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    last_time = time.time()
    start_time = time.time()

    logz.configure_output_dir(logdir)
    while True:
        if callback:
            callback(locals(), globals())
            
        if timesteps_so_far >= args.max_iters * 1e7:
            break

        if max_timesteps and timesteps_so_far >= max_timesteps:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        print("********** Iteration %i ************" % iters_so_far)
        if iters_so_far % 20 == 0:
            act = ActWrapper(model.pi)
            act.save(logdir)
            del act

        last_time = time_info('Other time cost ', last_time)

        weights = model.get_weights()
        ray.get([worker.set_weights.remote(weights) for worker in worker_list] + [worker.set_weights.remote(weights) for worker in eval_worker_list])

        last_time = time_info('Set weights time cost ', last_time)

        seg_list = ray.get([worker.run.remote() for worker in worker_list] + [worker.run.remote() for worker in eval_worker_list])
        seg = Worker.add_vtarg_and_adv(seg_list[:args.num_workers], gamma, lam) 
        eval_seg = Worker.add_vtarg_and_adv(seg_list[args.num_workers:], gamma, lam) 
        assert len(eval_seg) == len(seg)
        
        last_time = time_info('Collect data time cost ', last_time)

        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret),
                    shuffle=not model.pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]
        # update running mean/std for policy
        last_time = time_info('Prepare dataset ', last_time)

        if hasattr(model.pi, "ob_rms"):
            model.pi.ob_rms.update(ob)  # 标准化操作
        # set old parameter values to new parameter values
        model.assign_old_eq_new()

        last_time = time_info('Update ob rms and sync model ', last_time)
        
        for _ in range(optim_epochs):
            for batch in d.iterate_once(optim_batchsize):
                newlosses = model.update(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)

        last_time = time_info('Training ', last_time)

        lens, rews = seg["ep_lens"], seg["ep_rets"]
        eval_lens, eval_rews = eval_seg["ep_lens"], eval_seg["ep_rets"]
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        eval_lenbuffer.extend(eval_lens)
        eval_rewbuffer.extend(eval_rews)
        print(newlosses)
        logz.log_tabular("EpRewMean", np.mean(rewbuffer))
        logz.log_tabular("EpLenMean", np.mean(lenbuffer))
        logz.log_tabular("Eval EpRewMean", np.mean(eval_rewbuffer))
        logz.log_tabular("Eval EpLenMean", np.mean(eval_lenbuffer))
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logz.log_tabular("Iteration", timesteps_so_far)
        logz.log_tabular("Time cost so far", time.time() - start_time)
        logz.dump_tabular()

    print("Training Finished")    
    
    act = ActWrapper(model.pi)
    act.save(logdir)
    

def train(env_id, num_timesteps, seed, logdir, args=None, prefix=None):
    assert prefix is not None
    # from baselines.ppo1 import RLBlocks.mlp_policy as mlp_policy 
    U.make_session(num_cpu=1).__enter__()
    env = make_mujoco_env(env_id, seed, logdir, args)
    learn(env, max_timesteps=num_timesteps,
          timesteps_per_actorbatch=4096,
          clip_param=0.2, entcoeff=0.0,
          optim_epochs=10, optim_stepsize=5e-5, optim_batchsize=256,  # 3e-4
          gamma=0.95, lam=0.95, schedule='linear', logdir=logdir, args=args, prefix=prefix,
          )
    env.close()


def main(args):
    # args = mujoco_arg_parser().parse_args()
    env = env_name
    num_timesteps = int(args.total_iters * 1e7)
    seed = 0
    
    prefix = './Model_MidOverlap/data_noise{}'.format(args.noise_scale)
    
    logdir = env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    args.motion_file = 'Seg' + str(args.seg_index)
    logdir = os.path.join(prefix, args.motion_file, logdir)
    parentPath = os.path.join(prefix, args.motion_file)
    if not os.path.exists(parentPath):
        os.makedirs(parentPath)

    args.logdir = logdir
    train(env, num_timesteps=num_timesteps, seed=seed, logdir=logdir, args=args, prefix=prefix)


def get_args():
    parser = argparse.ArgumentParser(description="Leopard")

    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--seg-index', type=int, default=0, help='the index of the segment')

    '''hyper-parameters '''
    parser.add_argument('--noise-scale', type=float, default=0.0, help='learning rate')
    parser.add_argument('--max-iters', type=float, default=2.5, help='the max iteration, * 1e7')
    parser.add_argument('--total-iters', type=float, default=4, help='the max iteration, * 1e7')

    '''for env/action setting '''
    parser.add_argument('--min-start-frame', type=int, default=20, help='jump the first several frames')
    parser.add_argument('--duration', type=float, default=0.01667, help='duration')
    parser.add_argument('--rl-weight', type=float, default=0.25, help='learning rate')
    parser.add_argument('--random-scale', type=float, default=1.0, help='the partition we can start the initialization')
    parser.add_argument('--random-init', type=str, default='Arbitrary', choices=['Arbitrary', 'SegStart'], help='RFI')
    parser.add_argument('--seg-pairs', type=str, help='start segs and end segs')
    parser.add_argument('--pos-diff', type=float, default=1.0, help='the threshold of the position difference')
    parser.add_argument('--toe', type=float, default=1.5, help='the penalty added on toes')
    parser.add_argument('--enable-draw', dest='enable_draw', default=False, action='store_true')
    parser.add_argument('--time-shift', type=float, default=None, help='shift of initial time of env')

    '''for feature vector '''
    parser.add_argument('--phase-instr', type=str, default='normal', choices=['replace', 'normal'], help='how should we use the phases')
    parser.add_argument('--points', dest='points', default=False, action='store_true', help='select some points in the traj')
    parser.add_argument('--use-ort', dest='use_ort', default=False, action='store_true', help='use relative orientation?')
    parser.add_argument('--velocity', dest='velocity', default=False, action='store_true', help='use velocity?')
    parser.add_argument('--view-rad', type=int, default=6, help='the radius of the local view')
    parser.add_argument('--ds-step', type=int, default=20, help='downsampling step size')

    '''for neural network '''
    parser.add_argument('--network-type', type=str, choices=['normal', 'ensemble'], default='normal', help='the type of nn model')
    parser.add_argument('--num-layers', type=int, default=2, help='number of hidden layers in NN')
    parser.add_argument('--hid-size', type=int, default=128, help='number of hidden size')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu'], help='activation function')

    '''for algorithm '''
    parser.add_argument('--num-workers', type=int, default=16, help='number of workers to generate data')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    ray.init(object_store_memory=1 * 10**9)

    args = get_args()
    args.scale_norm_bit = 4 * args.view_rad // args.ds_step
    main(args)
