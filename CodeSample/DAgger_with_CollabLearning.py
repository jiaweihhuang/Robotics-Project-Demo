import gym
import pickle
import tensorflow as tf
import numpy as np
import argparse
import os
import ray
import RLBlocks.logz as logz
import time
import gym_ext_envs.gym_dm
import RLBlocks.tf_util as tf_util
import baselines.common.tf_util as U
import pybullet_data

from RLBlocks.util import ActWrapper
from collections import deque
from RLBlocks.DAggerPolicyClass import DAggerCLPolicy4Ray
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from RLBlocks.ReplayBuffer import TF_ReplayBuffer, TF_ReplayBuffer_with_Indices
from baselines.deepq.utils import load_state, save_state
from WorkerClass import Worker4DAgger, DatasetGenerateMultiSeg
from functools import partial
from copy import deepcopy
from baselines.common import Dataset

'''
python leopard_DeepMimic/DAgger_with_CollabLearning.py --random-scale 20.0 --points --velocity --seg-pairs "(0,7) (7,14)" --switch-shift 10 --view-rad 6 --ds-step 20 --random-init Arbitrary --use-ort --iter 50000 --loss-type AllMSE --use-std --num-learner 5 --cl-weight 0.1 --cl-type MSE --expert-dir MultiSegs

python leopard_DeepMimic/DAgger_with_CollabLearning.py --random-scale 1.0 --points --velocity --seg-pairs 0-7 --switch-shift 10 --view-rad 6 --ds-step 20 --random-init Arbitrary --use-ort --iter 50000 --loss-type AllMSE --use-std --num-learner 5 --cl-weight 0.1 --cl-type MSE --expert-dir MidOverlap
'''


config = {}

Overlap = 'Mid'
experts_model_path = '' 
mf_mid_path = 'LeopardDataMidOverlap'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gym_register_func = partial(
    gym.envs.register,
    id='dm_leopard-v1',
    entry_point='gym_ext_envs.gym_dm.envs:DMLeopardEnv',
    max_episode_steps=10000,)


'''
Remark:
For simplification, seg_start_frames, seg_end_frames, switch_policy_threshold always start from Seg0;
'''
def get_seg_config(args):
    end = args.seg_pairs[-1][-1] + 1
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
        seg_end_frames = seg_start_frames[1:] + [max(seg_start_frames) + 100]
    else:
        raise NotImplementedError

    switch_policy_threshold = seg_start_frames[1:end] + [max(seg_start_frames) + 100]  
    if args.switch_shift > 0:
        for i in range(len(switch_policy_threshold)):
            switch_policy_threshold[i] += args.switch_shift

    seg_start_frames = seg_start_frames[:end]
    seg_end_frames = seg_end_frames[:end]

    start_frame_index, end_frame_index = 0, seg_end_frames[-1]

    prefix = 'DAgger_SS{}_SE{}'.format(args.seg_pairs[0][0], args.seg_pairs[-1][-1])
    
    saver_base_path = 'Traj_start{}_end{}'.format(seg_start_frames[args.seg_pairs[0][0]], seg_end_frames[-1])
    
    motion_file = os.path.join('data', 'motions', 'leopard_retarget_motion.txt')    
    
    '''
    For multi segs (start_seg, end_seg), by default, we switch policy at switch_policy_threshold[end_seg]
    '''

    # seg_init_frames = [(seg_start_frames[s], seg_end_frames[s]) for s, _ in args.seg_pairs]
    # seg_init_frames = [(seg_start_frames[s], seg_end_frames[e]) for s, e in args.seg_pairs]
    if args.merge_long_segs:
        seg_init_frames = [(seg_start_frames[s], seg_end_frames[e]) for s, e in [(0, 7), (7, 14)]]
    else:
        seg_init_frames = [(seg_start_frames[s], seg_end_frames[e]) for s, e in args.seg_pairs]

    return {
        'prefix': prefix,
        'saver_base_path': saver_base_path,
        'motion_file': motion_file,
        'start_frame_index': start_frame_index, 
        'end_frame_index': end_frame_index,
        'seg_start_frames': seg_start_frames,
        'seg_end_frames': seg_end_frames,
        'switch_policy_threshold': switch_policy_threshold,
        'total_frames': seg_end_frames[-1] - seg_start_frames[args.seg_pairs[0][0]],
        'seg_init_frames': seg_init_frames,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    
    '''hyper-parameters '''
    parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
    parser.add_argument('--noise-scale', type=float, default=0.0, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=2000, help='learning rate')
    parser.add_argument('--eval-interval', type=int, default=500, help='learning rate')
    parser.add_argument('--buffer-size', type=int, default=50000, help='how many data generated for each segments')
    parser.add_argument('--new-data-size', type=int, default=8000, help='how many data generated for each segments')
    parser.add_argument('--loss-type', type=str, default='AllMSE', choices=['MLE', 'MSE', 'AllMSE', 'KL1', 'KL2', 'JS'], help='learning rate')
    parser.add_argument('--iter', type=int, default=100000, help='number of iterations')

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
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers to generate data')
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
    
    '''schedule sampling '''
    parser.add_argument('--SS', default=False, action='store_true', help='scheduled sampling')
    parser.add_argument('--ss-reject-prob', default=1.0, type=float, help='probability to not use schedule sampling')
    parser.add_argument('--ss-scale', default=2.0, type=float, help='random scale')
    parser.add_argument('--ss-lb', default=10, type=int, help='lower bound for the clipping')
    parser.add_argument('--ss-decay', default=False, action='store_true', help='scheduled sampling')
    parser.add_argument('--ss-increase', default=False, action='store_true', help='scheduled sampling')
    return parser.parse_args()


def make_env(args, phase_instr, seed=0):
    env = gym.make('dm_leopard-v1', motion_file=config['motion_file'], periodic=False, tm_args=args, phase_instr=phase_instr, seg_start_frames=config['seg_start_frames'], seg_end_frames=config['seg_end_frames'],
    seg_init_frames=config['seg_init_frames'])
    env.seed(seed)
    return env

def get_experts_paths(args):
    model_paths = []

    if args.merge_long_segs:
        for s, e in args.seg_pairs:
            model_path = None
            for f in os.listdir(os.path.join(experts_model_path, "SS{}_SE{}".format(s, e))):
                if not f.endswith('csv'):
                    model_path = os.path.join(experts_model_path, "SS{}_SE{}".format(s, e), f, 'model.pkl')
                    print(model_path)
                    break

            assert model_path is not None
            model_paths.append(model_path)
    else:
        for s, e in args.seg_pairs:
            model_path = None
            for f in os.listdir(os.path.join(experts_model_path, "Seg{}".format(s))):
                if not f.endswith('csv'):
                    model_path = os.path.join(experts_model_path, "Seg{}".format(s), f, 'model.pkl')
                    print(model_path)
                    break

            assert model_path is not None
            model_paths.append(model_path)

    return model_paths

'''
horizon = new_data_size // args.num_workers
'''
@ray.remote
class ParallelDAgger():
    def __init__(self, index, args, lr, expert_hid_size, model_paths, 
                horizon, buffer_path=None,
                ob_shape=None, action_space=None):
        set_global_seeds(index * 1000 + args.seed)      # to make sure the seed for each worker is different
        U.make_session(num_cpu=4).__enter__()    

        self.index = index
        self.args = deepcopy(args)
        self.rb = TF_ReplayBuffer_with_Indices.load(buffer_path, args.batch_size)
        
        self.main_graph = tf.get_default_graph()
        self.main_sess = tf.get_default_session()
        
        self.dagger_policy = DAggerCLPolicy4Ray(lr, args.batch_size, args=args, replayBuffer=self.rb, 
                                activation=args.activation, ob_shape=ob_shape, action_space=action_space, model_paths=model_paths, 
                                expert_hid_size=expert_hid_size,
                                seg_start_frames=config['seg_start_frames'], seg_end_frames=config['seg_end_frames'],
                                main_sess=self.main_sess, main_graph=self.main_graph)
        
        self.worker_list = []

        print('Build DAgger Worker')
        if args.merge_long_segs:
            experts_obs_dim = 264
        else:
            experts_obs_dim = 158
        for i in range(args.num_workers):
            worker = Worker4DAgger.remote(horizon, args=args, seg_config=config, 
                                gym_register_func=gym_register_func, index=i, 
                                name='BehaviorCloning/pi' if args.merge_long_segs else 'pi',
                                expert_hid_size=expert_hid_size,
                                env_func=partial(make_env, args, 'replace'), model_paths=model_paths, experts_obs_dim=experts_obs_dim) 
            self.worker_list.append(worker)

        ''' create saver '''
        policy_variables = self.dagger_policy.get_policy_variables()
        self.saver = tf.train.Saver(var_list=policy_variables, max_to_keep=1)
        self.main_graph.finalize()

    def generate_data(self):        
        weights = self.dagger_policy.get_weights()
        ray.get([worker.set_weights.remote(weights) for worker in self.worker_list])

        new_data = Worker4DAgger.merge_data(ray.get([worker.run_multi_seg.remote() for worker in self.worker_list]))

        if self.args.multi_explorer:
            self.setNewData2Buffer(new_data)
        
        return new_data

    def sample(self):
        self.obs, self.targets = self.dagger_policy.sample_data_from_rb()
        return self.obs

    def get_feedback(self, data_list):
        fb_dict = {}
        for j in range(len(data_list)):
            if j == self.index:
                fb_dict[j] = None
            else:
                fb_dict[j] = self.dagger_policy.get_pd(data_list[j])

        return fb_dict

    def setNewData2Buffer(self, data):
        with self.main_sess.as_default():
            self.rb.update(obs=data['obs'], targets=data['targets'], indices=data['indices'])
    
    def train(self, all_fb, cl_weight):
        if all_fb is not None and len(all_fb) > 1:
            fb = []
            for i in range(len(all_fb)):
                if i == self.index:
                    continue
                fb.append(all_fb[i][self.index])
            pd = sum(fb) / len(fb)
        else:
            pd = None

        return self.dagger_policy.train(self.obs, self.targets, pd, cl_weight)

    def save_model(self, logdir):
        self.saver.save(self.main_sess, os.path.join(logdir, 'Model{}'.format(self.index), 'model'))

    def restore_model(self, restore_dir):
        self.saver.restore(self.main_sess, restore_dir)



def main():
    ''' Revise variables according to args '''
    args = get_args()
    
    if 'MSE' in args.loss_type:
        args.use_std = True
    
    seg_pairs = []

    # interperate seg pairs
    global experts_model_path
    experts_model_path = "./PreTrain/" + args.expert_dir
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

    # get config
    global config
    config = get_seg_config(args)
    args.startFrame, args.endFrame = config['start_frame_index'], config['end_frame_index']


    seed = args.seed
    lr = args.lr
    set_global_seeds(seed)

    # define path of replay buffer & logger
    rb_file_name = 'ReplayBuffer_SS{}_SE{}.pickle'.format(args.seg_pairs[0][0], args.seg_pairs[-1][-1])
    rb_dir_name = 'ReplayBuffer_V{}_D{}'.format(args.view_rad, args.ds_step)

    if not args.merge_long_segs:
        rb_dir_name = 'SS_' + rb_dir_name       # SS is short for "short segs"
        
    logdir_prefix = ''
    if args.points:
        rb_dir_name = 'Points_' + rb_dir_name
        logdir_prefix = 'Points_' + logdir_prefix   
    if args.use_ort:
        logdir_prefix = 'UseOrt_' + logdir_prefix
    if args.velocity:
        logdir_prefix = 'Vel_' + logdir_prefix        
    if args.use_std:
        logdir_prefix = 'Std_' + logdir_prefix

    rb_file_name = logdir_prefix + rb_file_name
    buffer_path = os.path.join(rb_dir_name, rb_file_name)

    logdir = 'dm_leopard-v1' + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(logdir_prefix + config['prefix'], config['saver_base_path'], logdir)
    logz.configure_output_dir(logdir)
    

    U.make_session(num_cpu=4).__enter__()    
    
    main_graph = tf.get_default_graph()
    main_sess = tf.get_default_session()

    # generate dataset for imitation learning
    model_paths = get_experts_paths(args)

    args_for_BC = deepcopy(args)
    args_for_BC.random_init = 'Arbitrary'
    print(args.random_init)
    print(args_for_BC.random_init)
    if args.merge_long_segs:
        env_func = partial(make_env, args_for_BC, 'replace')
        name = 'BehaviorCloning/pi'
        args_for_BC.hid_size = 512
    else:
        env_func = partial(make_env, args_for_BC, 'replace')
        name = 'pi'
        args_for_BC.random_scale = 5.0
        args_for_BC.hid_size = 128

    with main_graph.as_default():
        with main_sess.as_default():
            if not os.path.exists(buffer_path):
                expert_indices = [i for i in range(len(args.seg_pairs))]
                if len(expert_indices) < 5:
                    parallel_scale = 5
                else:
                    parallel_scale = 1
                expert_indices = expert_indices * parallel_scale
                seed_list = [args.seed + i * 100 + 1000 for i in range(len(expert_indices))]

                if not os.path.exists(rb_dir_name):
                    os.makedirs(rb_dir_name)

                data_workers = [
                    DatasetGenerateMultiSeg.remote(
                        args_for_BC, index=expert_index, seed=seed, model_path=model_paths[expert_index], seg_start_frame=config['seg_start_frames'][args_for_BC.seg_pairs[expert_index][0]],
                        seg_end_frame=config['seg_end_frames'][args_for_BC.seg_pairs[expert_index][1]], switch_policy_threshold=config['switch_policy_threshold'][args_for_BC.seg_pairs[expert_index][1]], 
                        env_func=env_func, gym_register_func=gym_register_func, expert_hid_size=args_for_BC.hid_size, name=name,
                    ) for seed, expert_index in zip(seed_list, expert_indices)
                ]
                pretrain_datasets = ray.get([worker.generate_data_multiseg.remote(args_for_BC.buffer_size // len(expert_indices)) for worker in data_workers])
                obs = np.concatenate([data['obs'] for data in pretrain_datasets])
                targets = np.squeeze(np.concatenate([data['targets'] for data in pretrain_datasets]))
                indices = np.concatenate([data['indices'] for data in pretrain_datasets]).reshape([-1, 1])

                TF_ReplayBuffer_with_Indices.dump(buffer_path, obs, targets, indices)

    # Build DAgger policy
    gym_register_func()
    eval_env = make_env(args, 'replace')
    
    if args.merge_long_segs:
        expert_hid_size = 512
    else:
        expert_hid_size = 128
    dagger_policies = [
        ParallelDAgger.remote(index, args, lr, expert_hid_size, model_paths, 
                args.new_data_size // args.num_workers, buffer_path=buffer_path,
                ob_shape=eval_env.observation_space.shape, action_space=eval_env.action_space)
        for index in range(args.num_learner)
    ]

    
    # load pre-trained model if specified
    if args.restore_dir is not None:
        ray.get([learner.restore_model(args.restore_dir) for learner in dagger_policies])

    prt_interval = 50
    eval_interval = 500
    save_interval = 250

    ep_lens_list = [deque(maxlen=100) for _ in range(args.num_learner)]
    ep_rets_list = [deque(maxlen=100) for _ in range(args.num_learner)]
    

    # start the training
    for i in range(args.iter):
        # sample a batch
        data_list = ray.get(
            [learner.sample.remote() for learner in dagger_policies]
        )
        # get CL targets from other learner
        if args.num_learner > 1 and args.cl_weight >= 0.0:
            all_fb = ray.get(
                [learner.get_feedback.remote(data_list) for learner in dagger_policies]
            )
            # we do not use mutual learning until each learner is good enough
            cl_weight = args.cl_weight if i > 2000 else min(0.0, args.cl_weight)
        else:
            all_fb = None
            cl_weight = args.cl_weight
            
        loss_list = ray.get(
            [learner.train.remote(all_fb, cl_weight) for learner in dagger_policies]
        )
        
        # print training info
        if i > 0 and i % prt_interval == 0:
            with main_graph.as_default():
                with main_sess.as_default():
                    logz.log_tabular("Iteration", i)
                    
                    for j in range(len(loss_list)):                            
                        logz.log_tabular("Loss" + str(j), loss_list[j])
                        
                    logz.log_tabular("lr", args.lr)
                    if args.multi_explorer:
                        logz.log_tabular("Lens", [np.mean(ep_lens) for ep_lens in ep_lens_list])
                        logz.log_tabular("Rets", [np.mean(ep_rets) for ep_rets in ep_rets_list])
                    else:
                        logz.log_tabular("Lens", np.mean(ep_lens_list[0]))
                        logz.log_tabular("Rets", np.mean(ep_rets_list[0]))
                    logz.log_tabular("cl weight", cl_weight)
                    logz.dump_tabular()
        
        # save model
        if i > 0 and i % save_interval == 0:
            print('save to ', logdir)
            ray.get([learner.save_model.remote(logdir) for learner in dagger_policies])
            
        # evaluate the learner; generate new data and update the replayBuffer
        if i > 0 and i % eval_interval == 0:     # started to add new data
            if args.multi_explorer:
                info_list = ray.get(
                    [learner.generate_data.remote() for learner in dagger_policies]
                )
                for j in range(args.num_learner):
                    ep_lens_list[j].extend(info_list[j]['ep_len'])
                    ep_rets_list[j].extend(info_list[j]['ep_ret'])
            else:
                new_data = ray.get(dagger_policies[0].generate_data.remote())
                ray.get(
                    [learner.setNewData2Buffer.remote(new_data) for learner in dagger_policies]
                )

                ep_len, ep_ret = new_data['ep_len'], new_data['ep_ret']            

                ep_lens_list[0].extend(ep_len)
                ep_rets_list[0].extend(ep_ret)


if __name__ == '__main__':
    ray.init()
    main()