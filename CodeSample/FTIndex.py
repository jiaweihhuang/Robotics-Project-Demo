import gym
import RLBlocks.logz as logz
import os
import pickle
import tensorflow as tf
import baselines.common.tf_util as U
import numpy as np
import time
import argparse
import cloudpickle, tempfile, zipfile
import pybullet_data
import ray

from baselines.common import Dataset, explained_variance, zipsame
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from baselines.deepq.utils import load_state, save_state

from baselines.bench import Monitor
from baselines.common import set_global_seeds
from RLBlocks.normal_ppo import TF_Model
from RLBlocks.agent import Agent
from RLBlocks.util import ActWrapper
import RLBlocks.tf_util as tf_util
from WorkerClass import Worker
from functools import partial

from gym.envs.registration import register
from copy import deepcopy

'''
python leopard_DeepMimic/FTIndex.py --random-scale 20.0 --points --velocity --start-seg 0 --end-seg 14 --view-rad 6 --ds-step 20 --random-init Arbitrary --network-type gating --use-ort --model-dir
'''

import gym_ext_envs.gym_dm
gym.envs.register(
    id='dm_leopard-v1',
    entry_point='gym_ext_envs.gym_dm.envs:DMLeopardEnv',
    max_episode_steps=10000,
)
gym_register_func = partial(
    gym.envs.register,
    id='dm_leopard-v1',
    entry_point='gym_ext_envs.gym_dm.envs:DMLeopardEnv',
    max_episode_steps=10000,)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
env_name = 'dm_leopard-v1'

start_to_train_threshold = 15       # freeze pi but update vfn
Overlap = 'Mid'
config = {}


def get_seg_config(args):    
    start, end = 0, args.end_seg + 1

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
        seg_end_frames = seg_start_frames[1:] + [1e6]                             
    else:
        raise NotImplementedError    
    seg_start_frames = seg_start_frames[start:end]
    seg_end_frames = seg_end_frames[start:end]

    switch_policy_threshold = seg_start_frames[1:end] + [max(seg_start_frames) + 100]  
    if args.switch_shift > 0:
        for i in range(len(switch_policy_threshold)):
            switch_policy_threshold[i] += args.switch_shift

    return {
        'seg_start_frames': seg_start_frames,
        'seg_end_frames': seg_end_frames,
        # 'seg_init_frames': [(seg_start_frames[0], seg_end_frames[7]), (seg_start_frames[7], seg_end_frames[14])],
        'seg_init_frames': [(seg_start_frames[0], seg_end_frames[7])],
        'switch_policy_threshold': switch_policy_threshold,
    }


def get_session():
    return tf.get_default_session()

def make_mujoco_env(env_id, seed, logdir, args):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """    
    def get_file_prefix(name):
        if name.startswith('Seg'):
            return name + '_'
        return name

    motion_file = os.path.join('data', 'motions', 'leopard_retarget_motion.txt')

    set_global_seeds(seed)
    env = gym.make(env_id, motion_file=motion_file, seg_init_frames=args.config['seg_init_frames'],
                    seg_start_frames=args.config['seg_start_frames'], seg_end_frames=args.config['seg_end_frames'],
                    periodic=False, tm_args=args)
    
    # env = Monitor(env, logdir, allow_early_resets=True)
    env.seed(seed)
    
    return env


def get_model_paths(args):
    model_paths = []
    mid_model_path = "./PreTrain/MultiSegs"

    # if args.merge_long_segs:
    seg_pairs = [(0, 7), (7, 14)]
    for s, e in seg_pairs:
        model_path = None
        for f in os.listdir(os.path.join(mid_model_path, "SS{}_SE{}".format(s, e))):
            if not f.endswith('csv'):
                model_path = os.path.join(mid_model_path, "SS{}_SE{}".format(s, e), f, 'model.pkl')
                print(model_path)
                break

        assert model_path is not None
        model_paths.append(model_path)
    # else:
    #     for s, e in args.seg_pairs:
    #         model_path = None
    #         for f in os.listdir(os.path.join(mid_model_path, "Seg{}".format(s))):
    #             if not f.endswith('csv'):
    #                 model_path = os.path.join(mid_model_path, "Seg{}".format(s), f, 'model.pkl')
    #                 print(model_path)
    #                 break

    #         assert model_path is not None
    #         model_paths.append(model_path)

    return model_paths




def learn(env, timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          # annealing for stepsize parameters (epsilon and adam)
          schedule='constant', logdir=None,
          args=None, prefix=None,
          ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_shape = env.observation_space.shape
    ac_space = env.action_space

    env_func = partial(make_mujoco_env, env_id=env_name, logdir=args.logdir)

    
    if args.network_type == 'traj_norm':
        traj_data = env.get_leopard().get_traj_data()
        args.traj_info = {
            'mean': np.mean(traj_data, axis=0, keepdims=True),
            'std': np.std(traj_data, axis=0, keepdims=True)
        }


    print('the shape of ac_space is ', ac_space.shape)
    model = TF_Model(ob_shape, ac_space, clip_param, entcoeff, 
                prefix='BehaviorCloning', hid_size=args.hid_size, 
                traj_norm=args.traj_norm, scale_norm_bit=args.scale_norm_bit, 
                activation=args.activation, args=args, mask_pi=True, optim_stepsize=optim_stepsize)


    U.initialize()
    if 'gatingExpert' in args.network_type:
        model.load_experts_weights(get_model_paths(args), 512)

    
    model_dir_name = args.model_dir 
    restore_variables = []
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='BehaviorCloning/pi'):
        if 'Adam' not in v.name:
            restore_variables.append(v)
            print(v.name)
    print(restore_variables)

    base_motion_file = 'Traj_start{}_end{}'.format(args.config['seg_start_frames'][args.start_seg], args.config['seg_end_frames'][args.end_seg])
    base_model_dir = 'DAgger_SS{}_SE{}'.format(args.start_seg, args.end_seg)
    
    if args.points:
        base_model_dir = 'Points_' + base_model_dir
    if args.use_ort:
        base_model_dir = 'UseOrt_' + base_model_dir
    if args.velocity:
        base_model_dir = 'Vel_' + base_model_dir
    if args.transform:
        base_model_dir = 'Trans_' + base_model_dir
    if args.ankle:
        base_model_dir = 'Ankle_' + base_model_dir
    if args.use_std:
        base_model_dir = 'Std_' + base_model_dir

    if args.cartesian:
        logdir_prefix = 'C_' + logdir_prefix
    if args.zero_padding:
        logdir_prefix = 'ZP_' + logdir_prefix
    if args.zero_concat:
        logdir_prefix = 'ZC_' + logdir_prefix

    restore_dir = base_model_dir + '/' + base_motion_file + '/' + model_dir_name
    saver = tf.train.Saver(restore_variables)
    saver.restore(tf.get_default_session(), os.path.join(restore_dir, 'model'))
    
    worker_list = []
    if 'gating' in args.network_type:
        horizon = timesteps_per_actorbatch // args.num_workers // 2
    else:
        horizon = timesteps_per_actorbatch // args.num_workers
    for i in range(args.num_workers):
        worker = Worker.remote(horizon, args=args, switch_policy_threshold=config['switch_policy_threshold'],
                            hid_size=args.hid_size, num_hid_layers=2, gym_register_func=gym_register_func,
                            index=i, activation=args.activation, env_func=env_func)
        worker_list.append(worker)


    env_func = partial(make_mujoco_env, env_id=env_name, logdir=args.logdir)
    RSI_eval_worker_list = []
    NoRSI_eval_worker_list = []
    noise_eval_worker_list = []
    eval_timesteps_per_actorbatch = 4096
    eval_args = deepcopy(args)
    eval_args.random_scale = 1.0
    for i in range(args.num_eval_workers):
        worker = Worker.remote(eval_timesteps_per_actorbatch // args.num_eval_workers, args=eval_args,
                            hid_size=args.hid_size, num_hid_layers=2, gym_register_func=gym_register_func,
                            index=i, activation=args.activation, env_func=env_func)
        RSI_eval_worker_list.append(worker)

    eval_args = deepcopy(args)
    eval_args.random_scale = 20.0
    for i in range(args.num_eval_workers):
        worker = Worker.remote(eval_timesteps_per_actorbatch // args.num_eval_workers, args=eval_args,
                            hid_size=args.hid_size, num_hid_layers=2, gym_register_func=gym_register_func,
                            index=i, activation=args.activation, env_func=env_func)
        NoRSI_eval_worker_list.append(worker)

    eval_args = deepcopy(args)
    eval_args.random_scale = 1.0
    eval_args.noise_scale = 0.02
    for i in range(args.num_eval_workers):
        worker = Worker.remote(eval_timesteps_per_actorbatch // args.num_eval_workers, args=eval_args,
                            hid_size=args.hid_size, num_hid_layers=2, gym_register_func=gym_register_func,
                            index=i, activation=args.activation, env_func=env_func)
        noise_eval_worker_list.append(worker)
    all_eval_worker_list = RSI_eval_worker_list + NoRSI_eval_worker_list + noise_eval_worker_list


    # finalize graph
    tf.get_default_graph().finalize()

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

    rsi_eval_lenbuffer = deque(maxlen=30)
    rsi_eval_rewbuffer = deque(maxlen=30)
    norsi_eval_lenbuffer = deque(maxlen=30)
    norsi_eval_rewbuffer = deque(maxlen=30)
    noise_eval_lenbuffer = deque(maxlen=30)
    noise_eval_rewbuffer = deque(maxlen=30)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    eval_interval = 10
    logz.configure_output_dir(logdir)
    while True:
        if callback:
            callback(locals(), globals())

        if timesteps_so_far >= 2.5e7:
            break

        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        print("********** Iteration %i ************" % iters_so_far)
        if iters_so_far % 20 == 0:
            saver.save(tf.get_default_session(), save_path=os.path.join(logdir, 'model'))
        #     act = ActWrapper(model.pi)
        #     act.save(logdir)
        #     del act

        weights = model.get_weights()
        ray.get([worker.set_weights.remote(weights) for worker in worker_list])
        seg_list = ray.get([worker.run.remote() for worker in worker_list])

        seg = Worker.add_vtarg_and_adv(seg_list, gamma, lam) 
        
        if iters_so_far % eval_interval == 0:
            ray.get([worker.set_weights.remote(weights) for worker in all_eval_worker_list])
            info = ray.get([worker.run.remote() for worker in all_eval_worker_list])
            rsi_info, norsi_info, noise_info = info[:args.num_eval_workers], \
                        info[args.num_eval_workers:args.num_eval_workers*2], info[args.num_eval_workers*2:]
            assert len(rsi_info) == len(norsi_info) and len(norsi_info) == len(noise_info)

            for i in range(args.num_eval_workers):
                rsi_eval_lenbuffer.extend(rsi_info[i]['ep_lens'])
                rsi_eval_rewbuffer.extend(rsi_info[i]['ep_rets'])
                norsi_eval_lenbuffer.extend(norsi_info[i]['ep_lens'])
                norsi_eval_rewbuffer.extend(norsi_info[i]['ep_rets'])

                noise_eval_lenbuffer.extend(noise_info[i]['ep_lens'])
                noise_eval_rewbuffer.extend(noise_info[i]['ep_rets'])


        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        if args.cl_weight >= 0.0:
            pd = seg["pd"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        # standardized advantage function estimate 标准化操作
        atarg = (atarg - atarg.mean()) / atarg.std()
        if args.cl_weight >= 0.0:
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, index=seg['index'], pd=pd),
                    shuffle=not model.pi.recurrent)
        else:
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, index=seg['index']),
                    shuffle=not model.pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]
        # update running mean/std for policy
        if hasattr(model.pi, "ob_rms") and iters_so_far > start_to_train_threshold:
            if args.traj_norm is None:
                model.pi.ob_rms.update(ob)  # 标准化操作
            else:
                model.pi.ob_rms.update(ob[:, args.scale_norm_bit:])
        # set old parameter values to new parameter values
        model.assign_old_eq_new()
        for _ in range(optim_epochs):
            print('Epochs ', _)
            for batch in d.iterate_once(optim_batchsize):
                if args.cl_weight > 0.0:
                    newlosses = model.update_with_pd(
                        batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], batch["pd"],
                            cur_lrmult, float(iters_so_far > start_to_train_threshold)
                    )
                else:
                    newlosses = model.update(
                        batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, float(iters_so_far > start_to_train_threshold)
                    )

        lens, rews = seg["ep_lens"], seg["ep_rets"]
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        print('Losses', newlosses)
        print('Ep Lens ', lens)
        print('Ep rews ', rews)
        logz.log_tabular("EpRewMean", np.mean(rewbuffer))
        logz.log_tabular("EpLenMean", np.mean(lenbuffer))


        logz.log_tabular("RSI Eval EpRewMean", np.mean(rsi_eval_rewbuffer))
        logz.log_tabular("RSI Eval EpLenMean", np.mean(rsi_eval_lenbuffer))
        logz.log_tabular("NoRSI Eval EpRewMean", np.mean(norsi_eval_rewbuffer))
        logz.log_tabular("NoRSI Eval EpLenMean", np.mean(norsi_eval_lenbuffer))
        logz.log_tabular("Noise Eval EpRewMean", np.mean(noise_eval_rewbuffer))
        logz.log_tabular("Noise Eval EpLenMean", np.mean(noise_eval_lenbuffer))


        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logz.log_tabular("Iteration", timesteps_so_far)
        logz.dump_tabular()

        del seg

    print("Training Finished")
    


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def train(env_id, num_timesteps, seed, logdir, args=None, prefix=None):
    assert prefix is not None
    # from baselines.ppo1 import RLBlocks.mlp_policy as mlp_policy 
    U.make_session(num_cpu=1).__enter__()
    env = make_mujoco_env(env_id, seed, logdir, args)
    learn(env, max_timesteps=num_timesteps,
          timesteps_per_actorbatch=4096,
          clip_param=0.2, entcoeff=0.0,
          optim_epochs=args.optim_epochs, optim_stepsize=args.optim_stepsize, optim_batchsize=args.optim_batchsize,
        #   optim_epochs=10, optim_stepsize=5e-5, optim_batchsize=256,  # 3e-4
          gamma=0.95, lam=0.95, schedule='linear', logdir=logdir, args=args, prefix=prefix,
          )
    env.close()


def main(args):
    global config
    config = get_seg_config(args)

    args.startFrame = 0
    args.endFrame = config['seg_end_frames'][-1]

    args.config = config
    env = env_name
    num_timesteps = 4e7
    seed = 0

    args.seg_pairs = [(i, i) for i in range(args.start_seg, args.end_seg + 1)]
    
    prefix = './FTRSI_{}/data_noise{}'.format(args.urdf_style, args.noise_scale)
    
    logdir = env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(prefix, 'SS{}_SE{}'.format(args.start_seg, args.end_seg), logdir)
    parent_dir = os.path.join(prefix, 'SS{}_SE{}'.format(args.start_seg, args.end_seg))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    args.logdir = logdir
    train(env, num_timesteps=num_timesteps, seed=seed, logdir=logdir, args=args, prefix=prefix)


'''
python leopard_DeepMimic/FT_RSI_2Seg.py --random-scale 5.0 --noise-scale 0.02 --toe 0.5 --points --velocity --start-seg 7 --end-seg 14 

--network-type attention 
'''
def get_args():
    parser = argparse.ArgumentParser(description="Leopard")
    parser.add_argument('--num-class', type=int, default=0, help='how much motions need to be mixed')
    parser.add_argument('--seed', type=int, default=0, help='how much motions need to be mixed')
    parser.add_argument('--index', type=int, default=0, help='the index of the current motion')
    parser.add_argument('--duration', type=float, default=0.01667, help='duration of each frame')
    parser.add_argument('--rl-weight', type=float, default=0.25, help='the weights of RL')
    parser.add_argument('--random-scale', type=float, default=1.0, help='the partition we can start the initialization')
    parser.add_argument('--random-init', type=str, default='Arbitrary', 
                                        choices=['Arbitrary', 'False', 'SegStart', 'ContinueArb', 'ContinueSeg'], help='shall we use random init?')
    parser.add_argument('--hid-size', type=int, default=512, help='the hidden layer of NN')
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'tanh'], help='the activation of NN')
    parser.add_argument('--min-start-frame', type=int, default=20, help='the activation of NN')
    parser.add_argument('--noise-scale', type=float, default=0.0, help='the ratio of the original noise compare to the true value')
    parser.add_argument('--noise-type', type=str, default='rel', choices=['rel', 'abs'], 
                                help='relative noise or absolute noise')
    parser.add_argument('--pos-diff', type=float, default=0.3, help='the threshold of the position difference')
    parser.add_argument('--phase-instr', type=str, default='replace', choices=['replace', 'normal'], help='how should we use the phases')
    parser.add_argument('--view-rad', type=int, default=6, help='the radius of the local view')
    parser.add_argument('--ds-step', type=int, default=20, help='downsampling step size')
    parser.add_argument('--urdf-style', type=str, default='Standard', help='whether to use 0 mass')
    parser.add_argument('--toe', type=float, default=0.5, help='the penalty added on toes')
    parser.add_argument('--model-dir', type=str, help='the path of the relu model')
    parser.add_argument('--traj-norm', type=float, default=None, help='the normalize factor applying on traj-obs')
    
    parser.add_argument('--reset-shift', type=float, default=0.0, help='use together with Continue; normal value: 0.3')

    parser.add_argument('--enable-draw', dest='enable_draw', default=False, action='store_true')

    parser.add_argument('--num-workers', type=int, default=16, help='instructions about the penalty added on toes')
    parser.add_argument('--network-type', type=str, choices=['normal', 'attention', 'mask', 'gating', 'traj_norm', 'ensemble'], default='normal', help='the path of the relu model')

    parser.add_argument('--start-seg', type=int, default=None, help='downsampling step size')
    parser.add_argument('--end-seg', type=int, default=None, help='downsampling step size')
    parser.add_argument('--time-shift', type=float, default=None, help='downsampling step size')
    
    parser.add_argument('--opt-args', type=str, default=None, nargs='+', help='optim_epochs, optim_stepsize, optim_batchsize')

    parser.add_argument('--binary', dest='binary', default=False, action='store_true')

    parser.add_argument('--use-ort', dest='use_ort', default=False, action='store_true')
    parser.add_argument('--points', dest='points', default=False, action='store_true', help='select some points in the traj')

    parser.add_argument('--euler', dest='euler', default=False, action='store_true')
    parser.add_argument('--sin-cos', dest='sin_cos', default=False, action='store_true')
    parser.add_argument('--velocity', dest='velocity', default=False, action='store_true')
    parser.add_argument('--global-points', dest='global_points', default=False, action='store_true')
    parser.add_argument('--rel-ort', dest='rel_ort', default=False, action='store_true')
    
    parser.add_argument('--ankle', dest='ankle', default=False, action='store_true', help='use matrix?')
    parser.add_argument('--transform', dest='transform', default=False, action='store_true', help='use matrix?')
    parser.add_argument('--one-side', dest='one_side', default=False, action='store_true', help='use matrix?')
    parser.add_argument('--non-eq-interval', dest='non_eq_interval', default=False, action='store_true', help='use matrix?')
    parser.add_argument('--use-matrix', dest='use_matrix', default=False, action='store_true', help='use matrix?')

    
    parser.add_argument('--cartesian', dest='cartesian', default=False, action='store_true', help='use cartesian in states vector?')
    parser.add_argument('--zero-padding', dest='zero_padding', default=False, action='store_true', help='padding zeros?')
    parser.add_argument('--zero-caoncat', dest='zero_concat', default=False, action='store_true', help='padding zeros?')
    parser.add_argument('--train-expert', default=False, action='store_true', help='train expert or train gate only?')
    parser.add_argument('--freeze-gate', default=False, action='store_true', help='train expert or train gate only?')
    parser.add_argument('--traj-diff', default=False, action='store_true', help='train expert or train gate only?')
    parser.add_argument('--mixture-logstd', default=False, action='store_true', help='train expert or train gate only?')
    parser.add_argument('--num-experts', type=int, default=2, help='instructions about the penalty added on toes')
    parser.add_argument('--use-ae', dest='use_ae', default=False, action='store_true', help='use auto encoder to learn the feature?')
    parser.add_argument('--use-vae', dest='use_vae', default=False, action='store_true', help='use auto encoder to learn the feature?')

    parser.add_argument('--switch-shift', type=int, default=0, help='expert switch shift')
    parser.add_argument('--cl-weight', type=float, default=-1.0, help='the learning rate for kl with target')
    parser.add_argument('--im-weight', type=float, default=-1.0, help='the learning rate for kl with target')
    parser.add_argument('--cl-type', type=str, default='KL2', choices=['KL1', 'KL2', 'MSE', 'MLE'], help='the learning rate for kl with target')
    parser.add_argument('--use-std', default=False, action='store_true', help='use auto encoder to learn the feature?')
    parser.add_argument('--expert-type', type=str, default='large', choices=['large', 'small'], help='start segs and end segs')
    parser.add_argument('--num-eval-workers', default=8, type=int, help='how many experts in gating network???')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    ray.init()

    args = get_args()
    args.scale_norm_bit = 4 * args.view_rad // args.ds_step
    if args.traj_norm is not None:
        assert args.phase_instr == 'replace'

    if args.opt_args is not None:
        assert len(args.opt_args) == 3
        args.optim_epochs = int(args.opt_args[0])
        args.optim_stepsize = float(args.opt_args[1])
        args.optim_batchsize = int(args.opt_args[2])
    else:        
        args.optim_epochs, args.optim_stepsize, args.optim_batchsize = 10, 5e-5, 1024       # by default, we use 1024 (seems works better)
    main(args)
