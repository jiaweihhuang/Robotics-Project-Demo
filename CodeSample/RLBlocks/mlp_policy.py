from baselines.common.mpi_running_mean_std import RunningMeanStd, RunningMeanStd_Smooth
import baselines.common.tf_util as U
import tensorflow as tf
import gym
import time
import numpy as np
from baselines.common.distributions import make_pdtype



class BasicMLPClass(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def eval_mean_logstd(self, ob):
        return tf.get_default_session().run(
            [self.mean, self.logstd], {self.ob:ob[None]}
        )

    def eval_logstd(self):
        return tf.get_default_session().run(self.logstd)

    def reset_logstd(self, value):
        print("before reset, the logstd is ", tf.get_default_session().run(self.logstd))
        if type(value) is float:
            tf.get_default_session().run(tf.assign(self.logstd, tf.constant(value, dtype=tf.float32) * tf.ones_like(self.logstd, dtype=tf.float32)))
        else:
            tf.get_default_session().run(tf.assign(self.logstd, value.reshape(self.logstd.shape)))
        print("after reset, the logstd is ", tf.get_default_session().run(self.logstd))
        
    def rescale_logstd(self, scale=2.0):
        entropy = tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        print("before reset, the logstd is {}; Entropy is ".format(tf.get_default_session().run([self.logstd, entropy])))
        tf.get_default_session().run(tf.assign(self.logstd, self.logstd / scale))
        print("after reset, the logstd is {}; Entropy is ".format(tf.get_default_session().run([self.logstd, entropy])))

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        # print("ac1[0]",ac1[0])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    def get_pd(self, ob):
        return self.sess.run(self.pd.flat, feed_dict={self.ob: ob[None]})

    def set_sess_and_graph(self, sess, graph):
        self.sess = sess
        self.graph = graph


class MlpPolicy(BasicMLPClass):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            self.set_assign_list()
    
    def _build_normalizer(self):
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=self.ob_shape)
        self.obs_mean = self.ob_rms.mean
        self.obs_std = self.ob_rms.std

        
    ''' 
    Requirements:
    return vpred, mean, logstd
    '''
    def _build_network(self, trainable=False):       
        with tf.variable_scope('vf'):
            obz = tf.clip_by_value(
                (self.ob - self.obs_mean) / self.obs_std, -5.0, 5.0)
            self.obz = obz
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = self.act_func(tf.layers.dense(last_out, self.hid_size, name="fc%i" % (
                    i+1), kernel_initializer=U.normc_initializer(1.0), trainable=trainable))
            vpred = tf.layers.dense(
                last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0), trainable=trainable)[:, 0]

        hidden_layers = []
        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = tf.layers.dense(last_out, self.hid_size, name='fc%i' % (i+1), 
                                kernel_initializer=U.normc_initializer(1.0), trainable=trainable, activation=self.act_func)
                hidden_layers.append(last_out)

            mean = tf.layers.dense(last_out, self.pdtype.param_shape()[
                                    0]//2, name='final', kernel_initializer=U.normc_initializer(0.01), trainable=trainable)
                                    
            logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[
                                    0]//2], initializer=tf.zeros_initializer(), trainable=trainable)

        
        return vpred, mean, logstd, hidden_layers

    def rebuild_network_reuse_weights(self, ob, trainable):  
        with tf.variable_scope('vf', reuse=True):
            obz = tf.clip_by_value(
                (ob - self.obs_mean) / self.obs_std, -5.0, 5.0)
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = self.act_func(tf.layers.dense(last_out, self.hid_size, name="fc%i" % (
                    i+1), kernel_initializer=U.normc_initializer(1.0), trainable=trainable))
            vpred = tf.layers.dense(
                last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0), trainable=trainable)[:, 0]

        with tf.variable_scope('pol', reuse=True):
            last_out = obz
            for i in range(self.num_hid_layers):
                last_out = tf.layers.dense(last_out, self.hid_size, name='fc%i' % (i+1), 
                                kernel_initializer=U.normc_initializer(1.0), trainable=trainable, activation=self.act_func)
                
            mean = tf.layers.dense(last_out, self.pdtype.param_shape()[
                                    0]//2, name='final', kernel_initializer=U.normc_initializer(0.01), trainable=trainable)
                                    
            logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[
                                    0]//2], initializer=tf.zeros_initializer(), trainable=trainable)
        
        pdtype = make_pdtype(self.ac_space)

        pd = pdtype.pdfromflat(tf.concat([mean, mean * 0.0 + logstd], axis=1))
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, pd.sample(), pd.mode())

        return ac, vpred, stochastic, pd
        

    def _init(self, ob_shape, ac_space, hid_size, num_hid_layers, sess=None, gaussian_fixed_var=True, 
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

        if activation == 'tanh':
            self.act_func = tf.nn.tanh
        elif activation == 'relu':
            self.act_func = tf.nn.relu
        elif activation == 'elu':
            self.act_func = tf.nn.elu
        else:
            raise NotImplementedError

        if ob is None:
            self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_shape))
        elif ob == 'new_ph':
            self.ob = tf.placeholder(shape=[None] + list(ob_shape), dtype=tf.float32)
        else:
            self.ob = ob

        self._build_normalizer()

        self.pdtype = make_pdtype(ac_space)
        self.vpred, self.mean, self.logstd, self.hidden_layers = self._build_network(trainable=trainable)

        self.pd = self.pdtype.pdfromflat(tf.concat([self.mean, self.mean * 0.0 + self.logstd], axis=1))
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self.ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())

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

    def set_assign_list(self):
        self.policy_weights = []
        self.weights_ph = []
        self.assign_list = []
        self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)        
        for v in self.weights:
            print(v.name)
            v_ph = tf.placeholder(dtype=v.dtype, shape=v.shape)
            self.weights_ph.append(v_ph)
            self.assign_list.append(tf.assign(v, v_ph))        
            if 'vf' not in v.name:
                self.policy_weights.append(v)

    def get_weights(self):
        return self.sess.run(self.weights)

    def get_policy_weights(self):
        return self.sess.run(self.policy_weights)


    def set_weights(self, weights):
        assert len(weights) == len(self.assign_list)
        feed_dict = {}
        for (w_ph, w) in zip(self.weights_ph, weights):
            feed_dict[w_ph] = w            
        self.sess.run(
            self.assign_list, feed_dict=feed_dict
        )
        del feed_dict


