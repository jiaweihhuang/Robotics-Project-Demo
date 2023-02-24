import tensorflow as tf
import numpy as np
import time
import gym
import baselines.common.tf_util as U
from RLBlocks.util import ActWrapper
from baselines.common.distributions import make_pdtype
from RLBlocks.mlp_policy import BasicMLPClass
from baselines.common.mpi_running_mean_std import RunningMeanStd, RunningMeanStd_Smooth
from tensorflow.python.tools import inspect_checkpoint as chkp

class TrainGN(BasicMLPClass):
    def __init__(self, args, sess, trainable, num_experts,
                    expert_paths, horizon, env_func, gym_register_func, expert_hid_size=512,
                    name='GatingNetwork', stochastic=True):

        self.horizon = horizon
        self.sess = sess
        self.args = args
        self.stochastic = stochastic
        self.expert_hid_size = expert_hid_size
        
        # gym_register_func()
        self.env = env_func()
        self.ob_shape = self.env.observation_space.shape
        self.ac_space = self.env.action_space

        self.expert_paths = expert_paths
        self.name = name
        
        self.pi = GatingNetwork(self.name, self.ob_shape, self.ac_space, args.hid_size, args.num_layers, 
                    sess=sess, activation=args.activation, trainable=trainable, num_experts=num_experts)
        
        self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)))
        self.pi.load_experts_weights(expert_paths, expert_hid_size)
        
        
    def run(self):
        print(self.pi.act(False, np.array([i / 264 for i in range(264)])))
        print(self.pi.sess.run(self.pi.logstd))
        assert 0 == 1

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

        while True:
            if self.args.network_type == 'mask':
                ac, vpred = self.pi.act(self.stochastic, ob, int(self.env.get_frame() > threshold))
            else:
                ac, vpred = self.pi.act(self.stochastic, ob)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            
            obs.append(ob.copy())
            vpreds.append(vpred.copy())
            news.append(int(new))
            acs.append(ac.copy())
            indices.append(int(self.env.get_frame() > threshold))

            ob, rew, new, info = self.env.step(ac)
            
            rews.append(rew)

            cur_ep_ret += rew
            cur_ep_len += 1

            if new:
                print('start from ', self.env.cur_start_point, ' Rew is ', cur_ep_ret)
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                if t > self.horizon:
                    break
                ob = self.env.reset()
                print('Init from ', self.env.t / 0.01667)

            t += 1

        return {"ob": np.array(obs).astype(np.float64), "rew": np.array(rews), "vpred": np.array(vpreds), "new": np.array(news),
                "ac": np.array(acs), "index": np.array(indices).reshape([-1, 1]), "nextvpred": vpred * (1 - new), "ep_rets": ep_rets, "ep_lens": ep_lens}




class GatingNetwork(BasicMLPClass):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            self.set_assign_list()
    
    def _build_normalizer(self):
        with tf.variable_scope("obfilter"):
            ob_rms = RunningMeanStd(shape=self.ob_shape)
        return ob_rms

    def _init(self, ob_shape, ac_space, hid_size, num_hid_layers, sess=None, gaussian_fixed_var=True, args=None,
                    activation=None, trainable={'Gate': True, 'Expert': False, 'V': True}, mixture_logstd=False,
                    ob=None, index=None, num_experts=2):
        if isinstance(ob_shape, gym.spaces.Box):
            ob_shape = ob_shape.shape

        self.mixture_logstd = args.mixture_logstd

        assert trainable is not None
        if 'V' not in trainable.keys():
            trainable['V'] = trainable['Gate'] or trainable['Expert']

        self.sess = sess or tf.get_default_session()
        assert self.sess is not None, 'Session is None'

        self.hid_size = hid_size
        self.num_experts = args.num_experts
        self.num_hid_layers = num_hid_layers
        self.gaussian_fixed_var = gaussian_fixed_var
        self.ob_shape = ob_shape
        self.ac_space = ac_space
        self.args = args
        self.trainable = trainable

        if activation == 'tanh':
            self.act_func = tf.nn.tanh
        elif activation == 'relu':
            self.act_func = tf.nn.relu
        else:
            raise NotImplementedError

        if ob is None:
            self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_shape))
            self.index = U.get_placeholder(name="index", dtype=tf.float32, shape=[None, 1])
        elif ob == 'new_ph':
            self.ob = tf.placeholder(shape=[None] + list(ob_shape), dtype=tf.float32)
            self.index = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        else:
            self.ob = ob
            self.index = index

        self.expert_ob_rms_list = []
        self.expert_mean_list = []
        self.expert_logstd_list = []
        self.expert_obz_list = []

        self.pdtype = make_pdtype(ac_space)

        # gating networks
        self.ob_rms = self._build_normalizer()
        self.obz = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        self.vpred = self.build_value_network(self.obz, trainable=trainable['V'])
        self.gate = self.build_gating_network(self.obz, trainable=trainable['Gate'])

        self.gate_weights = tf.split(self.gate, self.num_experts, axis=1)        # get a list of tensors with shape [None, 1]
        # self.gate_weights = [0.0, 1.0]
        # self.gate_weights = [self.gate_weights[0] * 0.0 + 1.0, self.gate_weights[1] * 0.0]

        # experts networks
        self.mean, self.logstd = self.build_policy_network(self.gate_weights, trainable=trainable['Expert'], std=1.0)

        self.pd = self.pdtype.pdfromflat(tf.concat([self.mean, self.mean * 0.0 + self.logstd], axis=1))
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self.ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())


    ''' 
    Requirements:
    return vpred, mean, logstd
    '''
    def build_value_network(self, obz, trainable=False):       
        with tf.variable_scope('vf'):
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = self.act_func(tf.layers.dense(last_out, self.hid_size, name="fc%i" % (
                    i+1), kernel_initializer=U.normc_initializer(1.0), trainable=trainable))
            vpred = tf.layers.dense(
                last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0), trainable=trainable)[:, 0]
        return vpred

    def build_policy_network(self, gate_list, trainable=False, std=1.0):
        weights_list = []
        for i in range(self.num_experts):
            with tf.variable_scope('Expert{}'.format(i)): 
                if self.args.network_type == 'gating':      
                    obz = tf.clip_by_value(
                        (self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                    self.expert_obz_list.append(obz)
                else:                
                    assert 0 == 1
                    ob_rms = self._build_normalizer()
                    obz = tf.clip_by_value(
                        (self.ob - ob_rms.mean) / ob_rms.std, -5.0, 5.0)
                    self.expert_obz_list.append(obz)
                    self.expert_ob_rms_list.append(ob_rms)

                weights = []
                last_shape = self.expert_obz_list[i].shape[1]
                with tf.variable_scope('pol'):
                    for i in range(self.num_hid_layers):
                        out = np.random.randn(last_shape, self.hid_size).astype(np.float32)
                        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                        w = tf.Variable(initial_value=tf.constant(out), trainable=trainable, dtype=tf.float32)

                        out = np.random.randn(self.hid_size).astype(np.float32)
                        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                        b = tf.Variable(initial_value=np.random.randn(self.hid_size), trainable=trainable, dtype=tf.float32)
                        weights.append([w, b])       
                        last_shape = self.hid_size
                    
                    out = np.random.randn(last_shape, self.pdtype.param_shape()[0]//2).astype(np.float32)
                    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                    w = tf.Variable(initial_value=tf.constant(out), trainable=trainable, dtype=tf.float32)
                    
                    out = np.random.randn(self.pdtype.param_shape()[0]//2).astype(np.float32)
                    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                    b = tf.Variable(initial_value=tf.constant(out), trainable=trainable, dtype=tf.float32)
                    weights.append([w, b])
                    logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[
                                            0]//2], initializer=tf.zeros_initializer(), trainable=trainable, dtype=tf.float32)
                    weights.append(logstd)
            weights_list.append(weights)

        # mixture weights
        last_out = sum([self.expert_obz_list[j] * gate_list[j] for j in range(self.num_experts)])
        for i in range(self.num_hid_layers + 1):            
            last_out_list = [
                tf.matmul(last_out, weights_list[j][i][0]) + tf.expand_dims(weights_list[j][i][1], axis=0) for j in range(self.num_experts)
            ]
            last_out = sum([last_out_list[j] * gate_list[j] for j in range(self.num_experts)])
            if i < self.num_hid_layers:
                last_out = self.act_func(last_out)

        mean = last_out

        '''
        logstd = sum([weights_list[j][-1][tf.newaxis, :, :] * gate_list[j][:, :, tf.newaxis] for j in range(self.num_experts)])
        logstd = tf.squeeze(logstd, axis=1)
        print(mean.shape, logstd.shape)
        '''
        if self.mixture_logstd:
            logstd = sum([weights_list[j][-1] * gate_list[j] for j in range(self.num_experts)])
        else:
            logstd = weights_list[0][-1]
        
        self.all_logstd = [weights_list[j][-1] for j in range(self.num_experts)]
        return mean, logstd

    def reset_logstd(self, value):
        print("before reset, the logstd is ", tf.get_default_session().run(self.logstd))
        tf.get_default_session().run([
                tf.assign(self.all_logstd[j], tf.constant(value, dtype=tf.float32) * tf.ones_like(self.logstd, dtype=tf.float32))
            for j in range(self.num_experts)]
        )
        print("after reset, the logstd is ", tf.get_default_session().run(self.logstd))

    def build_gating_network(self, obz, trainable): # -> shape = [None, self.num_experts]
        with tf.variable_scope('gate'):
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = tf.layers.dense(last_out, self.hid_size, name='fc%i' % (i+1), 
                                kernel_initializer=U.normc_initializer(1.0), trainable=trainable, activation=self.act_func)
                
            logits = tf.layers.dense(last_out, self.num_experts, name='final', kernel_initializer=U.normc_initializer(0.01), trainable=trainable)
            gate = tf.nn.softmax(logits, axis=1)            

        self.gate_logits = logits
        return gate

    def update_ob_rms(self, obs):
        self.ob_rms.update(obs)

    def _act(self, stochastic, ob, *args):
        ac, vpred = self.sess.run([self.ac, self.vpred],
            feed_dict={
                self.ob: ob[None],
                self.stochastic: stochastic,
            }
        )
        return ac[0], vpred[0]

    def debug(self, stochastic, ob, *args):
        ac, mean, vpred = self.sess.run([self.ac, self.mean, self.vpred],
            feed_dict={
                self.ob: ob[None],
                self.stochastic: stochastic,
            }
        )
        print(mean)
        return ac[0], vpred[0]

    def act(self, stochastic, ob, *args):
        ac, vpred = self.sess.run([self.ac, self.vpred],
            feed_dict={
                self.ob: ob[None],
                self.stochastic: stochastic,
            }
        )
        return ac[0], vpred[0]

    def get_logits(self, ob):
        return self.sess.run([self.gate, self.gate_logits], {self.ob: ob[None]})

    def set_assign_list(self):
        self.weights_ph = []
        self.assign_list = []
        self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)        
        for v in self.weights:
            print(v.name)
            v_ph = tf.placeholder(dtype=v.dtype, shape=v.shape)
            self.weights_ph.append(v_ph)
            self.assign_list.append(tf.assign(v, v_ph))        

    def get_weights(self):
        return self.sess.run(self.weights)

    def set_weights(self, weights):
        assert len(weights) == len(self.assign_list)
        feed_dict = {}
        for (w_ph, w) in zip(self.weights_ph, weights):
            feed_dict[w_ph] = w            
        self.sess.run(
            self.assign_list, feed_dict=feed_dict
        )
        del feed_dict


    def load_experts_weights(self, expert_paths, expert_hid_size):
        print(self.scope)
        expert_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/Expert')

        weights_list = []
        for ep in expert_paths:
            graph = tf.Graph()
            sess = tf.Session(graph=graph)

            with graph.as_default():
                with sess.as_default():
                    act = ActWrapper.load(ep, self.ob_shape, self.ac_space, sess, name='BehaviorCloning/pi', hid_size=expert_hid_size, ob='new_ph')
            act.get_pi().set_sess_and_graph(sess, graph)

            weights = []
            # for w in act.get_pi().get_weights():
            for w in act.get_pi().weights:
                if 'vf' not in w.name:
                    weights.append(w)
            weights_list += act.get_pi().sess.run(weights)

        for i in range(len(weights_list)):
            print(weights_list[i].shape, expert_weights[i].shape)
        print(len(expert_weights), len(weights_list))

        assign_list = []
        expert_weights = []
        for i in range(self.num_experts):
            expert_weights += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/Expert{}'.format(i))            

        assert len(weights_list) == len(expert_weights)
        assign_list = [
            v.assign(w) for w, v in zip(weights_list, expert_weights)
        ]

        self.sess.run(assign_list)


