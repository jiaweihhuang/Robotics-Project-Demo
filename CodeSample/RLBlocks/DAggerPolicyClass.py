import tensorflow as tf
import os
import numpy as np
import time
import cloudpickle, tempfile, zipfile
from RLBlocks.mlp_policy import MlpPolicy
from RLBlocks.GatingNetwork import GatingNetwork
from RLBlocks.mlp_ensemble_policy import MlpEnsemblePolicy
import baselines.common.tf_util as U
from baselines.common import zipsame

'''
High level idea:

train (choice 1):
env1 -> generate data to replay buffer 1 -> train policy
env2 -> generate data to replay buffer 2 -> train policy

train (choice 2):
env1/env2 -> generate data to replay buffer -> train policy

test:
policy test in env1 -> revise one_hot/kronecker to env2 -> policy test in env2 -> ...
'''


def build_policy(policy_prefix, args, ob_shape, action_space, rb, trainable):
    if args.network_type == 'normal':
        policy = MlpPolicy(policy_prefix, 
                            ob_shape, action_space, trainable=trainable, ob=rb.obs_samples,
                            hid_size=args.hid_size, num_hid_layers=args.num_layers, activation=args.activation)
    elif args.network_type == 'ensemble':
        policy = MlpEnsemblePolicy(policy_prefix, 
                            ob_shape, action_space, trainable=trainable, ob=rb.obs_samples, args=args,
                            hid_size=args.hid_size, num_hid_layers=args.num_layers, activation=args.activation)
    elif 'gating' in args.network_type:
        policy = GatingNetwork(policy_prefix, 
                ob_shape, action_space, trainable={'Gate': True, 'Expert': args.train_expert}, args=args,
                ob=rb.obs_samples, index=rb.indices_samples,
                hid_size=args.hid_size, num_hid_layers=args.num_layers, activation=args.activation)
    else:
        raise NotImplementedError
    
    return policy

class DAggerPolicy4Ray(object):
    def __init__(self, lr, batch_size, *,
                        replayBuffer, 
                        activation=None, main_sess=None, seg_start_frames=None,
                        seg_end_frames=None, main_graph=None,
                        loss_type='MLE', trainable=True, eval_env=None, ob_shape=None, action_space=None, 
                        args=None, model_paths=None, expert_hid_size=None,
                        scope='BehaviorCloning'):
        self.lr = lr
        self.batch_size = batch_size
        self.replayBuffer = replayBuffer
        self.activation = activation
        self.seg_start_frames = seg_start_frames
        self.seg_end_frames = seg_end_frames

        self.network_type = args.network_type
        self.args = args

        self.main_sess = main_sess
        self.main_graph = main_graph
        assert main_sess is not None
        self.scope = scope
        self.policy_prefix = 'pi'
        self.loss_type = args.loss_type


        if eval_env is not None:
            ob_shape = eval_env.observation_space.shape
            action_space = eval_env.action_space


        with main_graph.as_default():
            with tf.variable_scope(self.scope):        
                self.policy = build_policy(self.policy_prefix, args, ob_shape, action_space, replayBuffer, trainable=trainable)

                self.targets = replayBuffer.targets_samples
    
                if self.network_type == 'ensemble':
                    self.loss_list = []
                    self.lr_decay_list = []
                    self.train_opt_list = []
                    for i in range(self.args.num_experts):
                        mean, logstd = self.policy.mean_list[i], self.policy.logstd_list[i]

                        loss = self.make_loss(mean, logstd, self.targets)

                        lr_decay = tf.placeholder(shape=[], dtype=tf.float32)
                        train_opt = tf.train.AdamOptimizer(learning_rate=lr_decay).minimize(loss)
                        self.loss_list.append(loss)
                        self.lr_decay_list.append(lr_decay)
                        self.train_opt_list.append(train_opt)
                else:
                    self.loss = self.make_loss(self.policy.mean, self.policy.logstd, self.targets)

                    if args.cl_weight >= 0.0:
                        self.pd_ph = tf.placeholder(tf.float32, [None, 24])
                        self.cl_loss = self.make_CL_loss(self.pd_ph, self.policy.mean, self.policy.logstd, cl_type=self.args.cl_type)
                        self.loss += self.args.cl_weight * self.cl_loss
                    
                    self.lr_decay = tf.placeholder(shape=[], dtype=tf.float32)
                    self.adam_op = tf.train.AdamOptimizer(learning_rate=self.lr_decay)
                    self.train_opt = self.adam_op.minimize(self.loss)

            self.init_variables()
            if 'gatingExpert' in args.network_type:
                self.policy.load_experts_weights(model_paths, expert_hid_size)

    def make_loss(self, mean, logstd, targets):
        if self.loss_type == 'MLE':
            target_mean = targets
            diff = (target_mean - mean)       
            loss = 0.5 * tf.reduce_sum(diff * diff / tf.exp(2 * logstd)) / tf.cast(tf.shape(target_mean)[0], tf.float32) + tf.reduce_sum(logstd) / tf.cast(tf.shape(logstd)[0], tf.float32)
        elif self.loss_type == 'MSE':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(tf.square(mean - target_mean))
        elif self.loss_type == 'AllMSE':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(tf.square(mean - target_mean)) + tf.reduce_mean(tf.square(logstd - target_logstd))
        elif self.loss_type == 'KL1':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(
                tf.reduce_sum(logstd + (tf.square(tf.exp(target_logstd)) + tf.square(mean - target_mean)) / (2.0 * tf.square(tf.exp(logstd))), axis=-1)
            )
        elif self.loss_type == 'KL2':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(
                tf.reduce_sum(target_logstd - logstd + (tf.square(tf.exp(logstd)) + tf.square(mean - target_mean)) / (2.0 * tf.square(tf.exp(target_logstd))), axis=-1)
            )
        elif self.loss_type == 'JS':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(
                tf.reduce_sum((tf.square(tf.exp(logstd)) + tf.square(mean - target_mean)) / (2.0 * tf.square(tf.exp(target_logstd)))\
                + (tf.square(tf.exp(target_logstd)) + tf.square(mean - target_mean)) / (2.0 * tf.square(tf.exp(logstd))), axis=-1)
            )
        else:
            raise NotImplementedError

        return loss

    def make_CL_loss(self, pd_ph, mean, logstd, cl_type):      
        target_mean, target_logstd = tf.split(pd_ph, 2, axis=1)
        target_std = tf.exp(target_logstd)
        std = tf.exp(logstd)

        if cl_type == 'KL1':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(logstd - target_logstd + (tf.square(target_std) + tf.square(mean - target_mean)) / (2.0 * tf.square(std)), axis=-1)
            )
        elif cl_type == 'KL2':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(target_logstd - logstd + (tf.square(std) + tf.square(mean - target_mean)) / (2.0 * tf.square(target_std)), axis=-1)
            )
        elif cl_type == 'MSE':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(mean - target_mean), axis=1)
            )
        elif cl_type == 'AllMSE':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(mean - target_mean), axis=1) \
                + tf.reduce_sum(tf.square(logstd - target_logstd), axis=1)
            )
        elif cl_type == 'MLE':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(mean - target_mean) / (2.0 * tf.square(std)), axis=1) + tf.reduce_sum(logstd, axis=1)
            )
        else:
            raise NotImplementedError
            
        return cl_loss

    def get_weights(self): 
        return self.policy.get_weights()

    def decay_lr(self, lr):
        self.lr = lr

    def init_variables(self):
        self.main_sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)))

    # for choice 2
    def train(self):
        with self.main_sess.as_default():
            if self.network_type == 'ensemble':
                loss_list = []
                for i in range(self.args.num_experts):
                    loss, obs, _ = self.main_sess.run(
                        [self.loss_list[i], self.replayBuffer.obs_samples, self.train_opt_list[i]],
                        feed_dict={self.lr_decay_list[i]: self.lr}    
                    )
                    self.policy.ob_rms_list[i].update(obs)
                    loss_list.append(loss)
                return loss_list, {'lr': self.lr}
            else:
                loss, obs, _ = self.main_sess.run(
                    [self.loss, self.replayBuffer.obs_samples, self.train_opt],
                    feed_dict={self.lr_decay: self.lr}    
                )
                if 'gating' in self.network_type:
                    self.policy.update_ob_rms(obs)
                else:
                    self.policy.ob_rms.update(obs)
                return loss, {'lr': self.lr}

    def get_action(self, *args):
        with self.main_graph.as_default():
            with self.main_sess.as_default():
                return self.policy.act(*args)

    def get_policy_variables(self):
        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + self.policy_prefix)
        policy_variables = []
        for v in all_variables:
            if 'Adam' in v.name:
                continue
            else:
                policy_variables.append(v)
        return policy_variables




class DAggerCLPolicy4Ray(object):
    def __init__(self, lr, batch_size, *,
                        replayBuffer, 
                        activation=None, main_sess=None, seg_start_frames=None,
                        seg_end_frames=None, main_graph=None,
                        loss_type='MLE', trainable=True, eval_env=None, ob_shape=None, action_space=None, 
                        args=None, model_paths=None, expert_hid_size=None,
                        scope='BehaviorCloning'):
        self.lr = lr
        self.batch_size = batch_size
        self.replayBuffer = replayBuffer
        self.activation = activation
        self.seg_start_frames = seg_start_frames
        self.seg_end_frames = seg_end_frames

        self.network_type = args.network_type
        self.args = args

        self.main_sess = main_sess
        self.main_graph = main_graph
        assert main_sess is not None
        self.scope = scope
        self.policy_prefix = 'pi'
        self.loss_type = args.loss_type


        if eval_env is not None:
            ob_shape = eval_env.observation_space.shape
            action_space = eval_env.action_space


        with main_graph.as_default():
            with tf.variable_scope(self.scope):        
                if self.args.network_type == 'ensembleGate':
                    assert replayBuffer is not None
                    self.policy = MlpEnsembleGatePolicy(self.policy_prefix, 
                                ob_shape, action_space, trainable=trainable, ob=self.replayBuffer.obs_samples, args=args,
                                hid_size=args.hid_size, num_hid_layers=args.num_layers, activation=args.activation)
                    self.targets = replayBuffer.targets_samples
                else:
                    self.policy = MlpPolicy(self.policy_prefix, 
                                ob_shape, action_space, trainable=trainable, ob='new_ph',
                                hid_size=args.hid_size, num_hid_layers=args.num_layers, activation=args.activation)
                    self.targets = tf.placeholder(shape=[None, 24], dtype=tf.float32)
                
    
                # if self.network_type == 'ensemble':
                #     self.loss_list = []
                #     self.lr_decay_list = []
                #     self.train_opt_list = []
                #     for i in range(self.args.num_experts):
                #         mean, logstd = self.policy.mean_list[i], self.policy.logstd_list[i]

                #         loss = self.make_loss(mean, logstd, self.targets)

                #         lr_decay = tf.placeholder(shape=[], dtype=tf.float32)
                #         train_opt = tf.train.AdamOptimizer(learning_rate=lr_decay).minimize(loss)
                #         self.loss_list.append(loss)
                #         self.lr_decay_list.append(lr_decay)
                #         self.train_opt_list.append(train_opt)
                # else:

                self.im_loss = self.make_loss(self.policy.mean, self.policy.logstd, self.targets)

                if self.args.network_type != 'ensembleGate' and args.cl_weight >= 0:
                    self.pd_ph = tf.placeholder(tf.float32, [None, 24])

                if self.args.network_type != 'ensembleGate' and args.cl_weight >= 0:
                    self.cl_weight = tf.placeholder(tf.float32, [])
                    self.cl_loss = self.make_CL_loss(self.pd_ph, self.policy.mean, self.policy.logstd, cl_type=self.args.cl_type)
                    self.loss = self.im_loss + self.cl_weight * self.cl_loss
                else:
                    self.loss = self.im_loss

                self.lr_decay = tf.placeholder(shape=[], dtype=tf.float32)
                self.adam_op = tf.train.AdamOptimizer(learning_rate=self.lr_decay)
                self.train_opt = self.adam_op.minimize(self.loss)

            self.init_variables()
            # if 'gatingExpert' in args.network_type:
            #     self.policy.load_experts_weights(model_paths, expert_hid_size)



    def make_loss(self, mean, logstd, targets):
        if self.loss_type == 'MLE':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            diff = (target_mean - mean)       
            loss = 0.5 * tf.reduce_sum(diff * diff / tf.exp(2 * logstd)) / tf.cast(tf.shape(target_mean)[0], tf.float32) + tf.reduce_sum(logstd) / tf.cast(tf.shape(logstd)[0], tf.float32)
        elif self.loss_type == 'MSE':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(tf.square(mean - target_mean))
        elif self.loss_type == 'AllMSE':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(tf.square(mean - target_mean)) + tf.reduce_mean(tf.square(logstd - target_logstd))
        elif self.loss_type == 'KL1':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(
                tf.reduce_sum(logstd + (tf.square(tf.exp(target_logstd)) + tf.square(mean - target_mean)) / (2.0 * tf.square(tf.exp(logstd))), axis=-1)
            )
        elif self.loss_type == 'KL2':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(
                tf.reduce_sum(target_logstd - logstd + (tf.square(tf.exp(logstd)) + tf.square(mean - target_mean)) / (2.0 * tf.square(tf.exp(target_logstd))), axis=-1)
            )
        elif self.loss_type == 'JS':
            target_mean, target_logstd = tf.split(targets, 2, axis=1)
            loss = tf.reduce_mean(
                tf.reduce_sum((tf.square(tf.exp(logstd)) + tf.square(mean - target_mean)) / (2.0 * tf.square(tf.exp(target_logstd)))\
                + (tf.square(tf.exp(target_logstd)) + tf.square(mean - target_mean)) / (2.0 * tf.square(tf.exp(logstd))), axis=-1)
            )
        else:
            raise NotImplementedError

        return loss

    def make_CL_loss(self, pd_ph, mean, logstd, cl_type):      
        target_mean, target_logstd = tf.split(pd_ph, 2, axis=1)
        target_std = tf.exp(target_logstd)
        std = tf.exp(logstd)

        if cl_type == 'KL1':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(logstd - target_logstd + (tf.square(target_std) + tf.square(mean - target_mean)) / (2.0 * tf.square(std)), axis=-1)
            )
        elif cl_type == 'KL2':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(target_logstd - logstd + (tf.square(std) + tf.square(mean - target_mean)) / (2.0 * tf.square(target_std)), axis=-1)
            )
        elif cl_type == 'MSE':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(mean - target_mean), axis=1)
            )
        elif cl_type == 'AllMSE':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(mean - target_mean), axis=1) \
                + tf.reduce_sum(tf.square(logstd - target_logstd), axis=1)
            )
        elif cl_type == 'MLE':
            cl_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(mean - target_mean) / (2.0 * tf.square(std)), axis=1) + tf.reduce_sum(logstd, axis=1)
            )
        else:
            raise NotImplementedError
            
        return cl_loss


    def sample_data_from_rb(self):
        return self.main_sess.run(
            [self.replayBuffer.obs_samples, self.replayBuffer.targets_samples]
        )

    def get_pd(self, ob):
        return self.main_sess.run(self.policy.pd.flat, feed_dict={self.policy.ob: ob})
    
    # note that in tf we can feed numpy to not only placeholder but also variables
    def eval_mean(self, ob):
        return self.main_sess.run(self.policy.mean, feed_dict={self.policy.ob: ob})

    def get_weights(self): 
        return self.policy.get_weights()

    def decay_lr(self, lr):
        self.lr = lr

    def init_variables(self):
        self.main_sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)))

    # for choice 2
    def train(self, obs, target, pd, cl_weight):
        with self.main_sess.as_default():
            if cl_weight >= 0.0 and pd is not None:
                loss, _ = self.main_sess.run(
                    [{'loss': self.im_loss, 'cl_loss': self.cl_loss}, self.train_opt],
                    feed_dict={self.lr_decay: self.lr, self.policy.ob: obs, 
                        self.targets: target, self.pd_ph: pd, self.cl_weight: cl_weight}    
                )
            else:
                loss, _ = self.main_sess.run(
                    [{'loss': self.im_loss}, self.train_opt],
                    feed_dict={self.lr_decay: self.lr, self.policy.ob: obs, 
                        self.targets: target}    
                )
            self.policy.ob_rms.update(obs)
            return loss
            
            
    def update(self, lr):
        with self.main_sess.as_default():
            loss, obs, _ = self.main_sess.run(
                [self.loss, self.replayBuffer.obs_samples, self.train_opt],
                feed_dict={self.lr_decay: lr}    
            )
            self.policy.ob_rms.update(obs)
            return loss

    def get_action(self, *args):
        with self.main_graph.as_default():
            with self.main_sess.as_default():
                return self.policy.act(*args)

    def get_policy_variables(self):
        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + self.policy_prefix)
        policy_variables = []
        for v in all_variables:
            if 'Adam' in v.name:
                continue
            else:
                policy_variables.append(v)
        return policy_variables

