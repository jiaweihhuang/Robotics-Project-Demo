import tensorflow as tf
import numpy as np
import time
import gym
import baselines.common.tf_util as U
from RLBlocks.util import ActWrapper
from baselines.common.distributions import make_pdtype
from RLBlocks.mlp_policy import BasicMLPClass, MlpPolicy
from baselines.common.mpi_running_mean_std import RunningMeanStd, RunningMeanStd_Smooth
from tensorflow.python.tools import inspect_checkpoint as chkp

'''
Multi policy network but only 1 value function
'''
class MlpEnsemblePolicy(MlpPolicy):
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

        
    ''' 
    Requirements:
    return vpred, mean, logstd
    '''
    def _build_value_network(self, obz, ob_rms, trainable=False):       
        with tf.variable_scope('vf'):
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = self.act_func(tf.layers.dense(last_out, self.hid_size, name="fc%i" % (
                    i+1), kernel_initializer=U.normc_initializer(1.0), trainable=trainable))
            vpred = tf.layers.dense(
                last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0), trainable=trainable)[:, 0]
        return vpred

    def _build_policy_network(self, obz, ob_rms, trainable=False):     
        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = tf.layers.dense(last_out, self.hid_size, name='fc%i' % (i+1), 
                                kernel_initializer=U.normc_initializer(1.0), trainable=trainable, activation=self.act_func)
                
            mean = tf.layers.dense(last_out, self.pdtype.param_shape()[
                                    0]//2, name='final', kernel_initializer=U.normc_initializer(0.01), trainable=trainable)
                                    
            logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[
                                    0]//2], initializer=tf.zeros_initializer(), trainable=trainable)
        
        return mean, logstd
        

    def _init(self, ob_shape, ac_space, hid_size, num_hid_layers, sess=None, gaussian_fixed_var=True, args=None,
                    activation=None, use_obs_norm=True, trainable=False, ob=None, index=None):
        if isinstance(ob_shape, gym.spaces.Box):
            ob_shape = ob_shape.shape

        assert trainable is not None

        self.sess = sess or tf.get_default_session()
        assert self.sess is not None, 'Session is None'

        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.gaussian_fixed_var = gaussian_fixed_var
        self.ob_shape = ob_shape
        self.ac_space = ac_space
        self.args = args
        self.num_experts = args.num_experts

        if activation == 'tanh':
            self.act_func = tf.nn.tanh
        elif activation == 'relu':
            self.act_func = tf.nn.relu
        else:
            raise NotImplementedError

        if ob is None:
            self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_shape))
        elif ob == 'new_ph':
            self.ob = tf.placeholder(shape=[None] + list(ob_shape), dtype=tf.float32)
        else:
            self.ob = ob

        self.ob_rms_list = []
        self.mean_list = []
        self.logstd_list = []
        self.pd_list, self.ac_list = [], []

        self.pdtype = make_pdtype(ac_space)
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())


        for i in range(self.num_experts):
            with tf.variable_scope('Expert{}'.format(i)):        
                ob_rms = self._build_normalizer()
                self.ob_rms_list.append(ob_rms) 
                obz = tf.clip_by_value(
                    (self.ob - ob_rms.mean) / ob_rms.std, -5.0, 5.0)  
                mean, logstd = self._build_policy_network(obz, ob_rms, trainable=trainable)
            
            self.mean_list.append(mean)
            self.logstd_list.append(logstd)

            pd = self.pdtype.pdfromflat(tf.concat([mean, mean * 0.0 + logstd], axis=1))            
            ac = U.switch(self.stochastic, pd.sample(), pd.mode())
            self.pd_list.append(pd)
            self.ac_list.append(ac)

            if i == 0:
                self.vpred = self._build_value_network(obz, ob_rms, trainable=trainable)     

        self.mean = sum(self.mean_list) / self.num_experts
        self.logstd = sum(self.logstd_list) / self.num_experts
        self.pd = self.pdtype.pdfromflat(tf.concat([self.mean, self.mean * 0.0 + self.logstd], axis=1))
        self.ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())

        self.mean_logstd_concat = tf.concat(self.mean_list + self.logstd_list, axis=1)

    def individual_act(self, ob, index):
        ac, vpred = self.sess.run([self.ac_list[index], self.vpred],
            feed_dict={
                self.ob: ob[None],
                self.stochastic: stochastic,
            }
        )
        return ac[0], vpred[0]

    def eval_mean_logstd(self, ob):
        self.sess.run(self.mean_logstd_concat, {self.ob: ob[None]})

    def _act(self, stochastic, ob, *args):
        ac, vpred = self.sess.run([self.ac, self.vpred],
            feed_dict={
                self.ob: ob[None],
                self.stochastic: stochastic,
            }
        )
        return ac[0], vpred[0]

    def get_obz(self, ob):
        return self.sess.run(self.obz, {self.ob: ob})

    def act(self, stochastic, ob, *args):
        ac, vpred = self.sess.run([self.ac, self.vpred],
            feed_dict={
                self.ob: ob[None],
                self.stochastic: stochastic,
            }
        )
        return ac[0], vpred[0]

