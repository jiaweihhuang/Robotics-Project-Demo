import ray
import time
import numpy as np
import tensorflow as tf
import RLBlocks.mlp_policy as mlp_policy 
from RLBlocks.GatingNetwork import GatingNetwork
from RLBlocks.mlp_ensemble_policy import MlpEnsemblePolicy

import gym
import gym_ext_envs.gym_dm
import os
import pybullet_data
from gym import spaces

from baselines.common import set_global_seeds
from baselines.bench import Monitor
from RLBlocks.util import ActWrapper
'''
Description:

When it's time to sample trajectories, the main policy copy it's weights to worker,
and then each work sample several trajectories synchronously.

'''


def get_observation_space(observation_dim):
    observation_high = np.array(
        [np.finfo(np.float32).max] * observation_dim)
    return spaces.Box(-observation_high, observation_high)

@ray.remote
class Worker(object):
    def __init__(self, horizon, stochastic=True, args=None, feature_dim=None,
                index=None, hid_size=128, activation=None, num_hid_layers=2, prefix=None,
                env_func=None, gym_register_func=None, expert_paths=None, switch_policy_threshold=None):
        if index > 0 and args.enable_draw:
            args.enable_draw = False

        gym_register_func()
        self.env = env_func(seed=index + args.seed * 100, args=args)
        
        set_global_seeds(10000 + index + args.seed)
        self.horizon = horizon
        self.stochastic = stochastic
        self.feature_dim = feature_dim
        self.args = args
        self.index = index

        
        if prefix is None:
            prefix = 'pi'
        if index is not None:
            pi_name = prefix + '_{}'.format(index)
        else:
            pi_name = prefix
            
        self.sess = tf.get_default_session() or tf.Session()
        
        ob_shape = self.env.observation_space.shape

        if args.network_type == 'normal':
            self.pi = mlp_policy.MlpPolicy(name=pi_name, ob_shape=ob_shape, 
                        ac_space=self.env.action_space, sess=self.sess, hid_size=hid_size, trainable=False,
                        num_hid_layers=num_hid_layers, activation=activation)
        elif args.network_type == 'ensemble':
            self.pi = MlpEnsemblePolicy(name=pi_name, ob_shape=ob_shape, args=args,
                        ac_space=self.env.action_space, sess=self.sess, hid_size=hid_size, trainable=False,
                        num_hid_layers=num_hid_layers, activation=activation)
        elif 'gating' in args.network_type:
            self.pi = GatingNetwork(name=pi_name, ob_shape=ob_shape, args=args,
                        ac_space=self.env.action_space, sess=self.sess, hid_size=hid_size, 
                        trainable={'Gate': False, 'Expert': False, 'V': False},
                        num_hid_layers=num_hid_layers, activation=activation)
        else:
            raise NotImplementedError


        self.init = tf.initialize_all_variables()
        self.sess.run(self.init)

        if hasattr(self.args, 'cl_weight') and (self.args.cl_weight >= 0.0 or self.args.im_weight >= 0.0):
            self.seg_start_frames = args.config['seg_start_frames']
            self.seg_end_frames = args.config['seg_end_frames']
            self.expert_type = args.expert_type
            if self.expert_type == 'small':
                self.load_expert_policies(expert_paths, 158, 128)
            else:
                self.load_expert_policies(expert_paths, 264, 512)

        if hasattr(self.args, 'SS') and self.args.SS:
            self.ss_lb = args.ss_lb
            self.ss_reject_prob = args.ss_reject_prob
            self.ss_scale = args.ss_scale

        tf.get_default_graph().finalize()

    @staticmethod
    def add_vtarg_and_adv(seg_list, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        seg = {}

        for name in ["ob", "rew", "vpred", "new", "ac", "index", "ep_rets", "ep_lens"]:
            seg[name] = np.concatenate([s[name] for s in seg_list])
            print(name, seg[name].shape)

        if "pd" in seg_list[0].keys():
            seg["pd"] = np.concatenate([s["pd"] for s in seg_list])

        seg["nextvpred"] = 0

        new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]
        print(seg["ob"].shape)

        return seg


    def load_expert_policies(self, model_paths, experts_obs_dim=158, expert_hid_size=128, experts_acts_dim=12, name='BehaviorCloning/pi'):
        self.graphs = []
        self.experts_list = []
        self.sess_list = []
        
        from gym import spaces        
        observation_high = np.array([np.finfo(np.float32).max] * experts_obs_dim)
        observation_space = spaces.Box(-observation_high, observation_high)
        action_space = spaces.Box(np.zeros(shape=[experts_acts_dim]), np.zeros(shape=[experts_acts_dim]))

        for i in range(len(model_paths)):
            g = tf.Graph()
            self.graphs.append(g)
            s = tf.Session(graph=g)
            self.sess_list.append(s)    

            path = model_paths[i]

            with g.as_default():
                with s.as_default():
                    act = ActWrapper.load(path, observation_space.shape, action_space, s, name=name, hid_size=expert_hid_size, ob='new_ph')
                    act.get_pi().set_sess_and_graph(s, g)
                    self.experts_list.append(act)
                    g.finalize()

    def set_weights(self, weights: dict):
        self.pi.set_weights(weights)
    
    def reset_reject_prob(self, reject_prob):
        self.ss_reject_prob = reject_prob

    def get_threshold(self, length):
        # with prob. self.ss_reject_prob, do not clip
        # with prob. 1 - self.ss_reject_prob, clip
        if np.random.rand() < self.ss_reject_prob:  
            return 0
        return max(0, min(length - self.ss_lb, int(np.random.rand() / self.ss_scale * length)))


    def run(self):
        t = 0
        ac = self.env.action_space.sample()  # not used, just so we have the datatype
        # print("in file agent.py ac.shape(env.action_space.sample())", np.array(ac).shape, ac)
        new = True  # marks if we're on first timestep of an episode
        ob = self.env.reset()

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        obs, rews, vpreds, news, acs = [], [], [], [], []
        indices = []

        threshold = 1090
        
        pocket = []


        while True:
            if self.args.network_type == 'mask':
                ac, vpred = self.pi.act(self.stochastic, ob, int(self.env.get_frame() > threshold))
            else:
                ac, vpred = self.pi.act(self.stochastic, ob)

            env_frame = int(self.env.get_frame())
            if 'gating' in self.args.network_type:
                if self.index == 0 and len(ep_lens) == 0 and env_frame % 10 == 0:
                    print('At frame ', env_frame, 'Gate weights and logits ', self.pi.get_logits(ob))
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value            

            pocket.append([ob.copy(), vpred.copy(), int(new), ac.copy(), int(self.env.get_frame() > threshold)])

            ob, rew, new, info = self.env.step(ac)

            pocket[-1].append(rew)
            
            cur_ep_ret += rew
            cur_ep_len += 1

            if new:
                # if SS, then randomly choose where to clip
                if hasattr(self.args, 'SS') and self.args.SS:
                    clip_start = self.get_threshold(length=len(pocket))
                else:
                    clip_start = 0
                for o_, v_, n_, a_, i_, r_ in pocket[clip_start:]:                    
                    obs.append(o_)
                    vpreds.append(v_)
                    news.append(n_)
                    acs.append(a_)
                    indices.append(i_)
                    rews.append(r_)

                pocket = []


                if len(ep_lens) <= 1 or cur_ep_len > 500 and self.index % 50 == 0:
                    print('start from ', self.env.cur_start_point, ' Rew is ', cur_ep_ret, ' Ep Len is ', cur_ep_len)
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                if len(obs) > self.horizon:
                    break
                ob = self.env.reset()
                if len(ep_lens) <= 1 or cur_ep_len > 500 and self.index % 50 == 0:
                    print('Init from ', self.env.t / 0.01667)

            t += 1

        if self.index == 1:
            print('Clip out {} samples'.format(t + 1 - len(obs)))

        return {"ob": np.array(obs).astype(np.float64), "rew": np.array(rews), "vpred": np.array(vpreds), "new": np.array(news),
                "ac": np.array(acs), "index": np.array(indices).reshape([-1, 1]), "nextvpred": vpred * (1 - new), "ep_rets": ep_rets, "ep_lens": ep_lens}


    def run_with_experts(self):
        t = 0
        ac = self.env.action_space.sample()  # not used, just so we have the datatype
        # print("in file agent.py ac.shape(env.action_space.sample())", np.array(ac).shape, ac)
        new = True  # marks if we're on first timestep of an episode
        ob = self.env.reset()

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        obs, rews, vpreds, news, acs = [], [], [], [], []
        pds = []
        indices = []

        threshold = 1090

        pocket = []
        
        expert_index = 0
        for i in range(len(self.switch_policy_threshold)):
            if expert_index + 1 < len(self.args.seg_pairs) and self.args.seg_pairs[expert_index+1][0] > 0 and self.env.get_frame() > self.switch_policy_threshold[self.args.seg_pairs[expert_index+1][0]-1]:
                expert_index += 1
            else:
                break

        while True:
            if self.args.network_type == 'mask':
                ac, vpred = self.pi.act(self.stochastic, ob, int(self.env.get_frame() > threshold))
            else:
                ac, vpred = self.pi.act(self.stochastic, ob)

            env_frame = int(self.env.get_frame())
            if 'gating' in self.args.network_type:
                if self.index == 0 and len(ep_lens) == 0 and env_frame % 10 == 0:
                    print('At frame ', env_frame, 'Gate weights and logits ', self.pi.get_logits(ob))
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            
            pocket.append([ob.copy(), vpred.copy(), int(new), ac.copy(), int(self.env.get_frame() > threshold)])
            

            with self.graphs[expert_index].as_default():
                with self.sess_list[expert_index].as_default():
                    if self.expert_type == 'small':
                        expert_ob = ob.copy()
                        expert_ob[0] = float(self.env.get_frame() - self.seg_start_frames[expert_index]) / float(self.seg_start_frames[expert_index] - self.seg_end_frames[expert_index])
                        expert_ob[1] = self.env.get_leopard().rootPosRel_backup
                    else:
                        expert_ob = ob
                    pd = self.experts_list[expert_index].get_pi().get_pd(expert_ob)
                    pocket[-1].append(np.squeeze(pd))

            ob, rew, new, info = self.env.step(ac)
            pocket[-1].append(rew)

            if expert_index + 1 < len(self.args.seg_pairs) and self.args.seg_pairs[expert_index+1][0] > 0 and self.env.get_frame() > self.switch_policy_threshold[self.args.seg_pairs[expert_index+1][0]-1]:
                expert_index += 1
                
            
            cur_ep_ret += rew
            cur_ep_len += 1

            if new:
                # if SS, then randomly choose where to clip
                if self.args.SS:
                    clip_start = self.get_threshold(length=len(pocket))
                else:
                    clip_start = 0
                for o_, v_, n_, a_, i_, p_, r_ in pocket[clip_start:]:                    
                    obs.append(o_)
                    vpreds.append(v_)
                    news.append(n_)
                    acs.append(a_)
                    indices.append(i_)
                    pds.append(p_)
                    rews.append(r_)

                pocket = []


                if len(ep_lens) <= 1 or cur_ep_len > 500 and self.index % 50 == 0:
                    print('start from ', self.env.cur_start_point, ' Rew is ', cur_ep_ret, ' Ep Len is ', cur_ep_len)
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                if len(obs) > self.horizon:
                    break
                ob = self.env.reset()
                if len(ep_lens) <= 1 or cur_ep_len > 500 and self.index % 50 == 0:
                    print('Init from ', self.env.t / 0.01667)

                expert_index = 0
                for i in range(len(self.switch_policy_threshold)):
                    if expert_index + 1 < len(self.args.seg_pairs) and self.args.seg_pairs[expert_index+1][0] > 0 and self.env.get_frame() > self.switch_policy_threshold[self.args.seg_pairs[expert_index+1][0]-1]:
                        expert_index += 1
                    else:
                        break

            t += 1

        if self.index == 1:
            print('Clip out {} samples'.format(t + 1 - len(obs)))

        return {"ob": np.array(obs).astype(np.float64), "rew": np.array(rews), "vpred": np.array(vpreds), "new": np.array(news),
                "ac": np.array(acs), "index": np.array(indices).reshape([-1, 1]), "nextvpred": vpred * (1 - new), "pd": np.array(pds),
                "ep_rets": ep_rets, "ep_lens": ep_lens}


@ray.remote
class DatasetGenerateMultiSeg(object):
    def __init__(self, args, index, seed, model_path, seg_start_frame, seg_end_frame, switch_policy_threshold,
                    env_func=None, gym_register_func=None, name='pi', AE_func=None, feature_dim=None, ae_path=None,
                    expert_hid_size=64):
        '''
        such env_func is 
        '''
        gym_register_func()
        
        set_global_seeds(10000 + seed + args.seed)

        self.env = env_func()
        self.env.seed(20000 + seed + args.seed)

        self.seed = seed
        self.index = index
        self.env.set_start_end_seg(args.seg_pairs[index][0], args.seg_pairs[index][1])
        self.stochastic = True

        self.args = args
        # self.expert_view_rad = args.view_rad
        # self.expert_ds_step = args.ds_step
        self.expert_view_rad = 6
        self.expert_ds_step = 20
        
        self.seg_start_frame = seg_start_frame
        self.seg_end_frame = seg_end_frame
        self.switch_policy_threshold = switch_policy_threshold

        self.sess = tf.Session()
        self.sess.__enter__()

        if args.merge_long_segs:
            expert_obs_shape = (156 + 2 * 6 * 9, )
        else:
            expert_obs_shape = (158,)

        with self.sess.as_default():
            self.act = ActWrapper.load(model_path, expert_obs_shape, self.env.action_space, self.sess, hid_size=expert_hid_size, name=name)
            self.act.get_pi().set_sess_and_graph(self.sess, tf.get_default_graph())

        tf.get_default_graph().finalize()

    def get_expert_obs(self):
        if self.args.merge_long_segs:
            mocap_data = self.env.get_mocap_data()
            cur_frame = self.env.get_leopard()._frame
            local_traj = []

            this_pose = mocap_data._motion_data['Frames'][cur_frame][1:4]
            this_ort = mocap_data._motion_data['Frames'][cur_frame][4:8]

            for i in range(1, self.expert_view_rad + 1):
                shift = i * self.expert_ds_step
                front_frame = mocap_data._motion_data['Frames'][max(0, cur_frame - shift)]
                rear_frame = mocap_data._motion_data['Frames'][min(mocap_data.NumFrames() - 1, cur_frame + shift)]


                front_pose = [front_frame[i + 1] - this_pose[i] for i in range(3)]
                rear_pose = [rear_frame[i + 1] - this_pose[i] for i in range(3)]
                
                front_ort = front_frame[4:8]
                rear_ort = rear_frame[4:8]
                    
                local_traj = rear_pose + rear_ort + local_traj + front_pose + front_ort
            
                if self.args.velocity:
                    front_next_frame = mocap_data._motion_data['Frames'][max(0, cur_frame - shift + 1)]
                    rear_next_frame = mocap_data._motion_data['Frames'][min(mocap_data.NumFrames() - 1, cur_frame + shift + 1)]

                    front_vxy = [(front_next_frame[i] - front_frame[i]) / self.env.duration for i in range(1, 3)]
                    rear_vxy = [(rear_next_frame[i] - rear_frame[i]) / self.env.duration for i in range(1, 3)]

                    local_traj = front_vxy + local_traj + rear_vxy

            ob4expert = local_traj + self.env.get_leopard().stateVector[-156:]
        else:
            ob4expert = self.env.get_leopard().stateVector[-158:].copy()
            ob4expert[0] = float(self.env.get_frame() - self.seg_start_frame) / float(self.seg_start_frame - self.seg_end_frame)
            ob4expert[1] = self.env.get_leopard().rootPosRel_backup
        return np.array(ob4expert)

    def generate_data_multiseg(self, sample_num):    
        ob = self.env.reset()        

        counter = 0
        rew_list = []
        total_rew = []
        ep_len = []
        ob_pocket, ac_pocket, i_pocket = [], [], []

        rb = {'obs': [], 'targets': [], 'indices': []}
        while True: 
            with self.sess.as_default():   
                # for cartesian
                expert_obs = self.get_expert_obs()
                ac = np.squeeze(self.act(self.stochastic, expert_obs)[0])
                if self.args.use_std:
                    target = self.act.get_pi().get_pd(expert_obs)
                else:
                    target = ac

            ob_pocket.append(ob)
            
            ac_pocket.append(target)
            i_pocket.append(self.index)

            ob, rew, new, _ = self.env.step(ac)
            counter += 1
            
            rew_list.append(rew)
            
            if self.env.get_frame() >= self.switch_policy_threshold:
                new = True

            if new:
                print('frame before reset ', self.env.get_frame())
                # only collect data if the traj is good enough
                if self.env.get_frame() < 0.3 or self.env.get_frame() > int(self.switch_policy_threshold - 10) or not self.args.merge_long_segs:
                    rb['obs'] += ob_pocket[:sample_num - len(rb['obs'])]
                    rb['targets'] += ac_pocket[:sample_num - len(rb['targets'])]
                    rb['indices'] += i_pocket[:sample_num - len(rb['indices'])]
                    if len(rb['obs']) >= sample_num:
                        assert len(rb['obs']) == sample_num
                        return rb

                print('switch_policy_threshold is ', self.switch_policy_threshold, 'Current Buffer Size is ', len(rb['obs']))
                print('This time start frame is ', self.env.cur_start_point)
                ob_pocket, ac_pocket, i_pocket = [], [], []

                total_rew.append(np.sum(rew_list))
                print('Average Rewards ', np.mean(total_rew))
                ep_len.append(counter)

                rew_list = []            
                counter = 0

                ob = self.env.reset()            


@ray.remote
class Worker4DAgger(object):
    @staticmethod
    def merge_data(data_list):
        data = {}
        data['obs'] = np.concatenate([data_list[i]['obs'] for i in range(len(data_list))])
        data['targets'] = np.squeeze(np.concatenate([data_list[i]['targets'] for i in range(len(data_list))]))
        data['indices'] = np.concatenate([data_list[i]['indices'] for i in range(len(data_list))]).reshape([-1, 1])
        for name in ['ep_len', 'ep_ret']:
            data[name] = []
            for i in range(len(data_list)):
                data[name] += data_list[i][name]
        return data

    def __init__(self, horizon, args=None, *, seg_config=None, prefix=None, index=None, name='pi', AE_func=None, feature_dim=None,
                env_func=None, gym_register_func=None, model_paths=None, ae_path=None,
                experts_obs_dim=158, expert_hid_size=128):
        if index > 0 and args.enable_draw:
            args.enable_draw = False
        
        set_global_seeds(10000 + index + args.seed)

        self.seg_start_frames = seg_config['seg_start_frames']
        self.seg_end_frames = seg_config['seg_end_frames']
        self.switch_policy_threshold = seg_config['switch_policy_threshold']
        self.seg_config = seg_config
        self.expert_hid_size = expert_hid_size
        self.feature_dim = feature_dim

        start_seg = args.start_seg 
        end_seg = args.end_seg

        # TODO, set start & end seg
        gym_register_func()
        self.env = env_func()
        self.env.seed(20000 + index + args.seed)
        
        self.horizon = horizon
        self.index = index
        self.stochastic = True
        self.args = args

        self.expert_view_rad = 6
        self.expert_ds_step = 20

        self.model_paths = model_paths
        
        if prefix is None:
            prefix = 'pi'
        if index is not None:
            pi_name = prefix + '_{}'.format(index)
        else:
            pi_name = prefix
            
        self.sess = tf.get_default_session() or tf.Session()
        self.sess.__enter__()

        ob_shape = self.env.observation_space.shape
        if 'gating' in args.network_type:              
            self.pi = GatingNetwork(pi_name, 
                    ob_shape, self.env.action_space, sess=self.sess, trainable={'Gate': False, 'Expert': False, 'V': False}, args=args,
                    hid_size=args.hid_size, num_hid_layers=args.num_layers, activation=args.activation)
            self.init = tf.initialize_all_variables()
            self.sess.run(self.init)
            if 'Expert' in args.network_type:
                self.pi.load_experts_weights(self.model_paths, self.expert_hid_size)
        else:
            if args.network_type == 'normal':
                self.pi = mlp_policy.MlpPolicy(name=pi_name, ob_shape=ob_shape, 
                            ac_space=self.env.action_space, sess=self.sess, hid_size=args.hid_size, 
                            num_hid_layers=args.num_layers, activation=args.activation)
            elif args.network_type == 'ensemble':
                self.pi = MlpEnsemblePolicy(name=pi_name, ob_shape=ob_shape, args=args,
                            ac_space=self.env.action_space, sess=self.sess, hid_size=args.hid_size, 
                            num_hid_layers=args.num_layers, activation=args.activation)
            else:
                raise NotImplementedError
            
            self.init = tf.initialize_all_variables()
            self.sess.run(self.init)

        self.load_expert_policies(experts_obs_dim=experts_obs_dim, name=name)

        tf.get_default_graph().finalize()

    def load_expert_policies(self, experts_obs_dim=158, experts_acts_dim=12, name='pi'):
        self.graphs = []
        self.experts_list = []
        self.sess_list = []
        
        from gym import spaces        
        observation_high = np.array([np.finfo(np.float32).max] * experts_obs_dim)
        observation_space = spaces.Box(-observation_high, observation_high)
        action_space = spaces.Box(np.zeros(shape=[experts_acts_dim]), np.zeros(shape=[experts_acts_dim]))

        for i in range(len(self.model_paths)):
            g = tf.Graph()
            self.graphs.append(g)
            s = tf.Session(graph=g)
            self.sess_list.append(s)    

            path = self.model_paths[i]

            with g.as_default():
                with s.as_default():
                    act = ActWrapper.load(path, observation_space.shape, action_space, s, name=name, hid_size=self.expert_hid_size, ob='new_ph')
                    act.get_pi().set_sess_and_graph(s, g)
                    self.experts_list.append(act)
                    g.finalize()
                    
    def get_expert_obs(self, expert_index):
        if self.args.merge_long_segs:

            mocap_data = self.env.get_mocap_data()
            cur_frame = self.env.get_leopard()._frame
            local_traj = []

            this_pose = mocap_data._motion_data['Frames'][cur_frame][1:4]
            this_ort = mocap_data._motion_data['Frames'][cur_frame][4:8]

            for i in range(1, self.expert_view_rad + 1):
                shift = i * self.expert_ds_step
                front_frame = mocap_data._motion_data['Frames'][max(0, cur_frame - shift)]
                rear_frame = mocap_data._motion_data['Frames'][min(mocap_data.NumFrames() - 1, cur_frame + shift)]

                front_pose = [front_frame[i + 1] - this_pose[i] for i in range(3)]
                rear_pose = [rear_frame[i + 1] - this_pose[i] for i in range(3)]
                
                front_ort = front_frame[4:8]
                rear_ort = rear_frame[4:8]
                    
                local_traj = rear_pose + rear_ort + local_traj + front_pose + front_ort
            
                # if self.args.velocity:
                front_next_frame = mocap_data._motion_data['Frames'][max(0, cur_frame - shift + 1)]
                rear_next_frame = mocap_data._motion_data['Frames'][min(mocap_data.NumFrames() - 1, cur_frame + shift + 1)]

                front_vxy = [(front_next_frame[i] - front_frame[i]) / self.env.duration for i in range(1, 3)]
                rear_vxy = [(rear_next_frame[i] - rear_frame[i]) / self.env.duration for i in range(1, 3)]

                local_traj = front_vxy + local_traj + rear_vxy

            ob4expert = local_traj + self.env.get_leopard().stateVector[-156:]
        else:
            ob4expert = self.env.get_leopard().stateVector[-158:].copy()
            ob4expert[0] = float(self.env.get_frame() - self.seg_start_frames[expert_index]) / float(self.seg_end_frames[expert_index] - self.seg_start_frames[expert_index])
            ob4expert[1] = self.env.get_leopard().rootPosRel_backup

        return np.array(ob4expert)

    '''
    Switch policy criterion:
    (1) expert_index + 1 < len(switch_policy_threshold)     (make sure this is not the last policy)
    (2) cur_frame > self.switch_policy_threshold[self.args.seg_pairs[expert_index+1][0]]    (it's time for the next policy to guide)

    if either is false, then do not switch policy
    '''
    def run_multi_seg(self):
        args = self.args

        ob = self.env.reset()        
        expert_index = 0
        cur_frame = self.env.get_frame()
        
        for i in range(len(self.switch_policy_threshold)):
            if expert_index + 1 < len(self.args.seg_pairs) and self.args.seg_pairs[expert_index+1][0] > 0 and self.env.get_frame() > self.switch_policy_threshold[self.args.seg_pairs[expert_index+1][0]-1]:
                expert_index += 1
            else:
                break

        expert = self.experts_list[expert_index]

        counter = 0
        rew_list = []
        total_rew = []
        ep_len = []
        pocket = []
        obs_buffer = []
        targets_buffer = []
        indices_buffer = []

        num_new_data = 0
        while True: 
            with self.sess_list[expert_index].as_default():   
                expert_obs = self.get_expert_obs(expert_index)
                target = np.squeeze(expert(self.stochastic, expert_obs)[0])
                if self.args.use_std:
                    target = expert.get_pi().get_pd(expert_obs)
            

            if self.args.network_type == 'mask':
                ac = np.squeeze(self.pi.act(self.stochastic, ob, expert_index)[0])                
            else:            
                ac = np.squeeze(self.pi.act(self.stochastic, ob)[0])     

            prev_ob = ob
            ob, rew, new, info = self.env.step(ac)

            pocket.append((prev_ob, target, expert_index))
            counter += 1
            
            rew_list.append(rew)
            if expert_index + 1 < len(self.args.seg_pairs) and self.args.seg_pairs[expert_index+1][0] > 0 and self.env.get_frame() > self.switch_policy_threshold[self.args.seg_pairs[expert_index+1][0]-1]:
                expert_index += 1
                expert = self.experts_list[expert_index]
            
            if self.env.get_frame() >= self.seg_config['end_frame_index']:
                new = True

            if new:
                total_rew.append(np.sum(rew_list))
                ep_len.append(counter)

                # by default, we discard the last few samples (we believe the quality is poor since the trajectories diverge from the RM too much)
                for (x, y, z) in pocket[:-5]:  
                    obs_buffer.append(x)
                    targets_buffer.append(y)
                    indices_buffer.append(z)

                if len(ep_len) % 3 == 0:
                    print('In this traj, StartPoint {}; Rews: {}; Lens: {}'.format(self.env.cur_start_point, np.sum(rew_list), counter))

                num_new_data += len(pocket)
                if num_new_data >= self.horizon:
                    print('Worker {} Finished Generating; New Data Added {}; Average Ret in this thread is {}'.format(self.index, len(obs_buffer), np.mean(total_rew)))
                    return {
                        'obs': obs_buffer,
                        'targets': targets_buffer,
                        'indices': indices_buffer,
                        'ep_len': ep_len,
                        'ep_ret': total_rew,
                    }

                # print('Current Buffer Size is ', len(targets_buffer))
                pocket = []


                rew_list = []            
                counter = 0

                ob = self.env.reset()            

                # print("mean ep len is ", np.mean(ep_len))
                # print("mean total reward is ", np.mean(total_rew))

                expert_index = 0
                
                for i in range(len(self.switch_policy_threshold)):
                    if expert_index + 1 < len(self.args.seg_pairs) and self.args.seg_pairs[expert_index+1][0] > 0 and self.env.get_frame() > self.switch_policy_threshold[self.args.seg_pairs[expert_index+1][0]-1]:
                        expert_index += 1
                    else:
                        break

                expert = self.experts_list[expert_index]

    
    def set_weights(self, weights: dict):
        self.pi.set_weights(weights)