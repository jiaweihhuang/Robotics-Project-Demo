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
from RLBlocks.CL_IM_PPO import CL_IM_Model
from RLBlocks.util import ActWrapper
import RLBlocks.tf_util as tf_util
from WorkerClass import Worker
from functools import partial
from copy import deepcopy

from gym.envs.registration import register

'''
python leopard_DeepMimic/IMCollabLearning.py --random-scale 20.0 --points --velocity --use-ort --seg-pairs "(0,7) (7,14)" --random-init Arbitrary --train-threshold 15 --opt-args 10 5e-5 4096 --cl-type MSE --cl-weight 0.1 --im-type MSE --im-weight 1.0 --num-learner 5 --restore-dir CL_MulExp
'''

import gym_ext_envs.gym_dm

gym_register_func = partial(
    gym.envs.register,
    id='dm_leopard-v1',
    entry_point='gym_ext_envs.gym_dm.envs:DMLeopardEnv',
    max_episode_steps=10000,)
gym_register_func()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
env_name = 'dm_leopard-v1'

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
        'seg_init_frames': [(seg_start_frames[0], seg_end_frames[7]), (seg_start_frames[7], seg_end_frames[14])],
        'switch_policy_threshold': switch_policy_threshold,
    }


def get_model_paths(args):
    model_paths = []
    mid_model_path = "./PreTrain/MultiSegs"

    if args.merge_long_segs:
        for s, e in args.seg_pairs:
            model_path = None
            for f in os.listdir(os.path.join(mid_model_path, "SS{}_SE{}".format(s, e))):
                if not f.endswith('csv'):
                    model_path = os.path.join(mid_model_path, "SS{}_SE{}".format(s, e), f, 'model.pkl')
                    print(model_path)
                    break

            assert model_path is not None
            model_paths.append(model_path)
    else:
        for s, e in args.seg_pairs:
            model_path = None
            for f in os.listdir(os.path.join(mid_model_path, "Seg{}".format(s))):
                if not f.endswith('csv'):
                    model_path = os.path.join(mid_model_path, "Seg{}".format(s), f, 'model.pkl')
                    print(model_path)
                    break

            assert model_path is not None
            model_paths.append(model_path)

    return model_paths


def make_mujoco_env(env_id, seed, logdir, args):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """    

    motion_file = os.path.join('data', 'motions', 'leopard_retarget_motion.txt')

    set_global_seeds(seed)
    env = gym.make(env_id, motion_file=motion_file, seg_init_frames=args.config['seg_init_frames'],
                    seg_start_frames=args.config['seg_start_frames'], seg_end_frames=args.config['seg_end_frames'],
                    periodic=False, tm_args=args)
    
    env.seed(seed)
    
    return env



@ray.remote
class ParallelCollabLearning():
    def __init__(self, args, index, ob_shape, ac_space, clip_param, entcoeff, 
                optim_stepsize,
                timesteps_per_actorbatch, env_func, gamma, lam):
        self.args = deepcopy(args)
        self.args.seed += index * 10
        self.index = index
        self.gamma = gamma
        self.lam = lam
        self.iters_so_far = 0
        
        # build model and restore variables
        U.make_session(num_cpu=1).__enter__()
        self.model = CL_IM_Model(ob_shape, ac_space, clip_param, entcoeff, 
                    prefix='BehaviorCloning', hid_size=args.hid_size, 
                    traj_norm=args.traj_norm, scale_norm_bit=args.scale_norm_bit, 
                    activation=args.activation, args=args, mask_pi=True, optim_stepsize=optim_stepsize)
        U.initialize()
        restore_variables = []
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='BehaviorCloning/pi'):
            if 'Adam' not in v.name:
                restore_variables.append(v)
                
        self.saver = tf.train.Saver(restore_variables)
        
        base_motion_file = 'Traj_start{}_end{}'.format(args.config['seg_start_frames'][args.start_seg], args.config['seg_end_frames'][args.end_seg])
        base_model_dir = 'DAgger_SS{}_SE{}'.format(args.start_seg, args.end_seg)
        
        if args.points:
            base_model_dir = 'Points_' + base_model_dir
        if args.use_ort:
            base_model_dir = 'UseOrt_' + base_model_dir
        if args.velocity:
            base_model_dir = 'Vel_' + base_model_dir
        if args.cl_weight >= 0.0 or args.im_weight >= 0.0:
            base_model_dir = 'Std_' + base_model_dir
                
        restore_dir = os.path.join(base_model_dir, base_motion_file, args.restore_dir)
        sub_model_dirs = []
        for d in os.listdir(restore_dir):
            if '.txt' in d:
                continue
            sub_model_dirs.append(d)
        sub_model_dir = sub_model_dirs[self.index]
        print(sub_model_dir)
        self.saver.restore(tf.get_default_session(), os.path.join(restore_dir, sub_model_dir, 'model'))


        self.worker_list = []
        for i in range(args.num_workers):
            worker = Worker.remote(timesteps_per_actorbatch // args.num_workers, 
                                args=args, switch_policy_threshold=config['switch_policy_threshold'],
                                hid_size=args.hid_size, num_hid_layers=2, gym_register_func=gym_register_func,
                                index=i, activation=args.activation, env_func=env_func, expert_paths=get_model_paths(args))
            self.worker_list.append(worker)


        # finalize graph
        tf.get_default_graph().finalize()

    def reset_worker_ss_prob(self, ss_reject_prob):
        ray.get([worker.reset_reject_prob.remote(ss_reject_prob) for worker in self.worker_list])


    def generate_data(self):
        weights = self.model.get_weights()
        ray.get([worker.set_weights.remote(weights) for worker in self.worker_list])
        seg_list = ray.get([worker.run_with_experts.remote() for worker in self.worker_list])

        self.seg = Worker.add_vtarg_and_adv(seg_list, self.gamma, self.lam) 

        return self.seg['ob']

    def get_feedback(self, obs_list):
        fb_dict = {}
        for i in range(len(obs_list)):
            if i == self.index:  
                # this is your data, no need feedback
                fb_dict[i] = None
                continue
            fb_dict[i] = self.model.get_pd(obs_list[i])
        return fb_dict

    def update_with_cl(self, all_fb, cur_lrmult, iters_so_far, cl_weight, im_weight, optim_batchsize, optim_epochs):
        # average over other learner's feedback
        seg = self.seg
        pds_list = []
        for fb in all_fb:
            if fb[self.index] is not None:
                pds_list.append(fb[self.index])
        avg_pd = sum(pds_list) / len(pds_list)


        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        # standardized advantage function estimate 标准化操作
        atarg = (atarg - atarg.mean()) / atarg.std()
        
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, index=seg['index'], im_pd=seg['pd'], cl_pd=avg_pd),
                shuffle=True)
                
        optim_batchsize = optim_batchsize or ob.shape[0]
        # update running mean/std for policy
        if iters_so_far > self.args.train_threshold:
            self.model.pi.ob_rms.update(ob) 
            
        # set old parameter values to new parameter values
        self.model.assign_old_eq_new()
        for _ in range(optim_epochs):
            print('Epochs ', _)
            for batch in d.iterate_once(optim_batchsize):
                newlosses = self.model.update_with_CL_IM(
                    batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], 
                        cur_lrmult, cl_weight=cl_weight, im_weight=im_weight, im_pds=batch["im_pd"], cl_pds=batch["cl_pd"],
                        pi_lr_mask=float(iters_so_far > args.train_threshold),
                )

        return_info = {
            'ep_lens': seg['ep_lens'],
            'ep_rets': seg['ep_rets'],
            'loss': newlosses
        }
        return return_info


    def update(self, cur_lrmult, iters_so_far, im_weight, optim_batchsize, optim_epochs):
        seg = self.seg

        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        # standardized advantage function estimate 标准化操作
        atarg = (atarg - atarg.mean()) / atarg.std()
        
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, index=seg['index'], im_pd=seg['pd']),
                shuffle=True)

        optim_batchsize = optim_batchsize or ob.shape[0]
        # update running mean/std for policy
        if iters_so_far > self.args.train_threshold:
            self.model.pi.ob_rms.update(ob) 
            
        # set old parameter values to new parameter values
        self.model.assign_old_eq_new()
        for _ in range(optim_epochs):
            print('Epochs ', _)
            for batch in d.iterate_once(optim_batchsize):
                newlosses = self.model.update_with_IM(
                    batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], 
                        cur_lrmult, im_weight=im_weight, im_pds=batch["im_pd"],
                        pi_lr_mask=float(iters_so_far > args.train_threshold),
                )

        return_info = {
            'ep_lens': seg['ep_lens'],
            'ep_rets': seg['ep_rets'],
            'loss': newlosses
        }
        return return_info


    def save_model(self, logdir):
        self.saver.save(tf.get_default_session(), save_path=os.path.join(logdir, 'Model{}'.format(self.index), 'model'))


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
    learner_list = [
        ParallelCollabLearning.remote(args, index, 
                ob_shape, ac_space, clip_param, entcoeff, optim_stepsize,
                timesteps_per_actorbatch=timesteps_per_actorbatch, env_func=env_func,
                gamma=gamma, lam=lam) 
        for index in range(args.num_learner)
    ]


    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = [deque(maxlen=100) for _ in range(args.num_learner)]  # rolling buffer for episode lengths
    rewbuffer = [deque(maxlen=100) for _ in range(args.num_learner)]  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    logz.configure_output_dir(logdir)
    while True:
        if callback:
            callback(locals(), globals())

        if timesteps_so_far >= args.max_iters * 1e7:
            break

        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if args.cl_decay:
            cl_weight = max(1.0, args.cl_weight * (1.0 - float(timesteps_so_far) / max_timesteps))
        else:
            cl_weight = args.cl_weight

        if args.im_decay:
            im_weight = max(1.0, args.im_weight * (1.0 - float(timesteps_so_far) / max_timesteps))
        else:
            im_weight = args.im_weight

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        print("********** Iteration %i ************" % iters_so_far)
        if iters_so_far % 20 == 0:
            ray.get(
                [learner.save_model.remote(logdir) for learner in learner_list]
            )


        if args.ss_decay:
            ss_reject_prob = args.ss_reject_prob * max(float(timesteps_so_far) / max_timesteps, 0)
            ray.get([learner.reset_worker_ss_prob.remote(ss_reject_prob) for learner in learner_list])
        elif args.ss_increase:
            ss_reject_prob = args.ss_reject_prob * max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            ray.get([learner.reset_worker_ss_prob.remote(ss_reject_prob) for learner in learner_list])

        obs_list = ray.get(
            [learner.generate_data.remote() for learner in learner_list]
        )
        if args.cl_weight >= 0.0:
            fb_from_other_learner = ray.get(
                [learner.get_feedback.remote(obs_list) for learner in learner_list]
            )
            info_list = ray.get(
                [learner.update_with_cl.remote(fb_from_other_learner, cur_lrmult, iters_so_far, cl_weight, im_weight, optim_batchsize, optim_epochs) for learner in learner_list]
            )
        else:
            info_list = ray.get(
                [learner.update.remote(cur_lrmult, iters_so_far, im_weight, optim_batchsize, optim_epochs) for learner in learner_list]
            )

        

        for i in range(args.num_learner):
            lenbuffer[i].extend(info_list[i]["ep_lens"])
            rewbuffer[i].extend(info_list[i]["ep_rets"])
        print('Ep Lens ', info_list[0]["ep_lens"])
        print('Ep rews ', info_list[0]["ep_rets"])
        logz.log_tabular("EpRewMean", [np.mean(rewb) for rewb in rewbuffer])
        logz.log_tabular("EpLenMean", [np.mean(lenb) for lenb in lenbuffer])

        loss_names = info_list[0]['loss'].keys()
        for n in loss_names:
            logz.log_tabular(n, [info_list[i]['loss'][n] for i in range(args.num_learner)])
        timesteps_so_far += sum(info_list[0]["ep_lens"])
        iters_so_far += 1
        logz.log_tabular("cur_lrmult", cur_lrmult)
        logz.log_tabular("CL weights", cl_weight)
        logz.log_tabular("IM weights", im_weight)
        logz.log_tabular("Iteration", timesteps_so_far)
        logz.dump_tabular()


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
          gamma=0.95, lam=0.95, schedule=args.schedule, logdir=logdir, args=args, prefix=prefix,
          )
    env.close()


def main(args):
    env = env_name
    num_timesteps = args.total_iters * 1e7
    seed = 0

    seg_pairs = []

    if '-' in args.seg_pairs:
        args.merge_long_segs = False
        start, end = args.seg_pairs.split('-')
        for i in range(int(start), int(end) + 1):
            seg_pairs.append((i, i))
    else:
        args.merge_long_segs = True
        for s in args.seg_pairs.split(')')[:-1]:
            start_seg, end_seg = s.split('(')[-1].split(',')
            seg_pairs.append((int(start_seg), int(end_seg)))
        for i in range(len(seg_pairs) - 1):
            assert seg_pairs[i][0] < seg_pairs[i][1], 'Start seg should always be smaller than end seg'
            assert seg_pairs[i][1] >= seg_pairs[i+1][0], 'There must exist overlap between segs'
            assert seg_pairs[i][0] < seg_pairs[i+1][0], 'No redundancy'

    print('Input seg_pairs are ', seg_pairs)
    args.seg_pairs = seg_pairs
    args.start_seg = seg_pairs[0][0]
    args.end_seg = seg_pairs[-1][-1]
    
    global config
    config = get_seg_config(args)

    args.startFrame = 0
    args.endFrame = config['seg_end_frames'][-1]

    args.config = config

    prefix = './FT_CL_IM_{}/data_noise{}'.format(args.urdf_style, args.noise_scale)
    
    logdir = env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = 'IM{}_CL{}_'.format(args.im_type, args.cl_type) + logdir
    logdir = os.path.join(prefix, 'SS{}_SE{}'.format(args.start_seg, args.end_seg), logdir)
    parent_dir = os.path.join(prefix, 'SS{}_SE{}'.format(args.start_seg, args.end_seg))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    args.logdir = logdir
    train(env, num_timesteps=num_timesteps, seed=seed, logdir=logdir, args=args, prefix=prefix)


'''
python leopard_DeepMimic/FTCL.py --random-scale 20.0 --toe 0.5 --points --velocity --use-ort --start-seg 7 --end-seg 14 
--network-type attention 
'''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    
    '''hyper-parameters '''
    parser.add_argument('--max-iters', type=float, default=2.5, help='the max iteration for training')
    parser.add_argument('--total-iters', type=float, default=4, help='the max iteration before lr decay to 0, * 1e7')
    parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
    parser.add_argument('--noise-scale', type=float, default=0.0, help='learning rate')
    parser.add_argument('--schedule', default='linear', type=str, choices=['linear', 'constant'], help='decay cl weight??')
    parser.add_argument('--train-threshold', type=int, default=15, help='when to start to update policy weights')
    parser.add_argument('--opt-args', type=str, default=None, nargs='+', help='optim_epochs, optim_stepsize, optim_batchsize')

    '''for env/action setting '''
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
    parser.add_argument('--phase-instr', type=str, default='replace', choices=['replace', 'normal'], help='how should we use the phases')
    parser.add_argument('--points', dest='points', default=False, action='store_true', help='select some points in the traj')
    parser.add_argument('--use-ort', dest='use_ort', default=False, action='store_true', help='use relative orientation?')
    parser.add_argument('--velocity', dest='velocity', default=False, action='store_true', help='use velocity?')
    parser.add_argument('--view-rad', type=int, default=6, help='the radius of the local view')
    parser.add_argument('--ds-step', type=int, default=20, help='downsampling step size')

    '''for neural network '''
    parser.add_argument('--network-type', type=str, choices=['normal', 'gating', 'ensemble'], 
            default='normal', help='the type of nn model')
    parser.add_argument('--num-layers', type=int, default=2, help='number of hidden layers in NN')
    parser.add_argument('--hid-size', type=int, default=512, help='number of hidden size')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu'], help='activation function')

    '''for algorithm '''
    parser.add_argument('--num-learner', type=int, default=1, help='how many learner involved in mutual learning?')
    parser.add_argument('--num-workers', type=int, default=16, help='number of workers to generate data')
    parser.add_argument('--switch-shift', type=int, default=0, help='add-on value for switch threshold')
    parser.add_argument('--expert-dir', type=str, default='MultiSegs', choices=['MultiSegs', 'RSI_Noise', 'MidOverlap'], help='model dir of experts')
    parser.add_argument('--restore-dir', type=str, default=None, help='file name of replaybuffer.pickle')

    '''for gating network '''
    parser.add_argument('--train-expert', default=False, action='store_true', help='train expert or train gate only?')
    parser.add_argument('--mixture-logstd', default=False, action='store_true', help='train expert or train gate only?')
    parser.add_argument('--num-experts', default=2, type=int, help='how many experts in gating network???')

    '''for collaborative learning '''
    parser.add_argument('--use-std', default=False, action='store_true', help='remove reference motion')
    parser.add_argument('--cl-weight', type=float, default=-1.0, help='the learning rate for kl with target')
    parser.add_argument('--cl-type', type=str, default='MSE', choices=['KL1', 'KL2', 'MSE', 'MLE'], help='the learning rate for kl with target')
    parser.add_argument('--multi-explorer', default=False, action='store_true', help='each learner interact individually?')
    
    '''for imitation learning '''
    parser.add_argument('--im-weight', type=float, default=-1.0, help='the learning rate for kl with target')
    parser.add_argument('--im-decay', default=False, action='store_true', help='decay cl weight??')
    parser.add_argument('--im-type', type=str, default='MSE', choices=['KL1', 'KL2', 'MSE', 'AllMSE', 'MLE'], help='the learning rate for kl with target')
    parser.add_argument('--expert-type', type=str, default='large', choices=['large', 'small'], help='which type of expert?')

    '''schedule sampling '''
    parser.add_argument('--SS', default=False, action='store_true', help='scheduled sampling')
    parser.add_argument('--ss-reject-prob', default=1.0, type=float, help='probability to not use schedule sampling')
    parser.add_argument('--ss-scale', default=2.0, type=float, help='random scale')
    parser.add_argument('--ss-lb', default=10, type=int, help='lower bound for the clipping')
    parser.add_argument('--ss-decay', default=False, action='store_true', help='scheduled sampling')
    parser.add_argument('--ss-increase', default=False, action='store_true', help='scheduled sampling')

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
        args.optim_epochs, args.optim_stepsize, args.optim_batchsize = 10, 5e-5, 4096       # by default, we use 1024 (seems works better)
    main(args)
