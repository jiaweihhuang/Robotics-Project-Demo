import tensorflow as tf
import pickle
import numpy as np
import ray
import os
import baselines.common.tf_util as U
import RLBlocks.mlp_policy as mlp_policy 
from RLBlocks.GatingNetwork import GatingNetwork
from RLBlocks.mlp_ensemble_policy import MlpEnsemblePolicy
from baselines.common import zipsame
from baselines.common.normal_adam import NormalAdam


def policy_fn(name, ob_shape, ac_space, hid_size=64, num_hid_layers=2, activation=None, 
                trainable=True, args=False):
    if 'gating' in args.network_type:
        if trainable == True:
            trainable={'Gate': True, 'Expert': args.train_expert}
        else:
            trainable={'Gate': False, 'Expert': False}
        return GatingNetwork(name=name, ob_shape=ob_shape, ac_space=ac_space, args=args,
                        hid_size=hid_size, num_hid_layers=num_hid_layers, num_experts=args.num_experts,
                        activation=activation, trainable=trainable)
    elif 'ensemble' in args.network_type:
        return MlpEnsemblePolicy(name=name, ob_shape=ob_shape, ac_space=ac_space, args=args,
                        hid_size=hid_size, num_hid_layers=num_hid_layers,
                        activation=activation, trainable=trainable)
    else:
        return mlp_policy.MlpPolicy(name=name, ob_shape=ob_shape, ac_space=ac_space,
                        hid_size=hid_size, num_hid_layers=num_hid_layers,
                        activation=activation, trainable=trainable)


class TF_Model(object):
    def __init__(self, ob_shape, ac_space, clip_param, entcoeff,
                prefix=None, hid_size=64, activation=None, optim_stepsize=5e-5, 
                num_hid_layers=2, mask_pi=False, args=None):
        sess = tf.get_default_session()
        if prefix is not None:
            pi_name = '{}/pi'.format(prefix)
        else:
            pi_name = 'pi'

        pi = policy_fn(pi_name, ob_shape, ac_space, hid_size=hid_size,
                        activation=activation, num_hid_layers=num_hid_layers, args=args,
                        trainable=True)
        oldpi = policy_fn("oldpi", ob_shape, ac_space, hid_size=hid_size, 
                        activation=activation, num_hid_layers=num_hid_layers, args=args,
                        trainable=False)

        atarg = tf.placeholder(name='adv', dtype=tf.float32, shape=[None])  # Target advantage function
        # Target value function, empirical return
        ret = tf.placeholder(name='ret', dtype=tf.float32, shape=[None])
        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
        clip_param = clip_param * lrmult
        # ob = U.get_placeholder_cached(name="ob")
        ob = pi.ob
        ac = pi.pdtype.sample_placeholder([None])
        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-entcoeff) * meanent

        self.pi_lr_mask = tf.placeholder(shape=[], dtype=tf.float32)

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
        surr1 = ratio * atarg  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(
            ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #

        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
        total_loss = self.pi_lr_mask * (pol_surr + pol_entpen) + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        self.pd_ph = tf.placeholder(tf.float32, [None, 24])
        self.target_mean, self.target_logstd = tf.split(self.pd_ph, 2, axis=1)
        self.target_std = tf.exp(self.target_logstd)
        self.logstd = pi.logstd
        self.mean = pi.mean
        self.std = tf.exp(pi.logstd)

        # TODO, omit zero gradient terms
        if hasattr(args, 'im_weight') and args.im_weight >= 0.0:
            self.im_weight = tf.placeholder(tf.float32, [])
            kl1_loss = tf.reduce_mean(
                tf.reduce_sum(self.logstd - self.target_logstd + (tf.square(self.target_std) + tf.square(self.mean - self.target_mean)) / (2.0 * tf.square(self.std)), axis=-1)
            )
            kl2_loss = tf.reduce_mean(
                tf.reduce_sum(self.target_logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - self.target_mean)) / (2.0 * tf.square(self.target_std)), axis=-1)
            )
            mse_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.mean - self.target_mean), axis=1)
            )
            allmse_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.mean - self.target_mean), axis=1) \
                + tf.reduce_sum(tf.square(self.logstd - self.target_logstd), axis=1)
            )
            mle_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.mean - self.target_mean) / (2.0 * tf.square(self.std)), axis=1) + tf.reduce_sum(self.logstd, axis=1)
            )

            if args.im_type == 'KL1':
                cl_loss = kl1_loss
                loss_names += ['KL1 loss (CL)', 'KL2 loss', 'MSE loss', 'AllMSE loss', 'MLE loss']
            elif args.im_type == 'KL2':
                cl_loss = kl2_loss
                loss_names += ['KL1 loss', 'KL2 loss (CL)', 'MSE loss', 'AllMSE loss', 'MLE loss']
            elif args.im_type == 'MSE':
                cl_loss = mse_loss
                loss_names += ['KL1 loss', 'KL2 loss', 'MSE loss (CL)', 'AllMSE loss', 'MLE loss']
            elif args.im_type == 'AllMSE':
                cl_loss = allmse_loss
                loss_names += ['KL1 loss', 'KL2 loss', 'MSE loss', 'AllMSE loss (CL)', 'MLE loss']
            elif args.im_type == 'MLE':
                cl_loss = mle_loss
                loss_names += ['KL1 loss', 'KL2 loss', 'MSE loss', 'AllMSE loss', 'MLE loss (CL)']
            else:
                raise NotImplementedError

            losses += [kl1_loss, kl2_loss, mse_loss, allmse_loss, mle_loss]

            total_loss += self.im_weight * cl_loss * self.pi_lr_mask

        '''
        tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
        '''


        with tf.variable_scope('Adam'):
            update_op = tf.train.AdamOptimizer(lrmult * optim_stepsize).minimize(total_loss)

        if mask_pi:
            lossandgrad = U.function(
                [ob, ac, atarg, ret, lrmult, self.pi_lr_mask], losses + [update_op])
        else:
            lossandgrad = U.function(
                [ob, ac, atarg, ret, lrmult], losses + [update_op])


        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in
                                                        zipsame(oldpi.get_variables(), pi.get_variables())])
        
        compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

        self.pi = pi
        self.oldpi = oldpi
        self.lossandgrad = lossandgrad
        self.compute_losses = compute_losses
        self.assign_old_eq_new = assign_old_eq_new
        self.loss_names = loss_names
        #
        self.sess = sess
        self.ob = ob
        self.ac = ac
        self.atarg = atarg
        self.ret = ret
        self.lrmult = lrmult
        self.total_loss = total_loss
        self.update_op = update_op
        self.losses = losses

    def load_experts_weights(self, model_paths, expert_hid_size):        
        self.pi.load_experts_weights(model_paths, expert_hid_size)

    def get_weights(self):
        return self.pi.get_weights()

    def update(self, obs, acs, advs, values, lrmult, pi_lr_mask=1.0):
        td_map = {self.ob: obs, self.ac: acs, self.atarg: advs,
                  self.ret: values, self.lrmult: lrmult, self.pi_lr_mask: pi_lr_mask}
        losses, _ = self.sess.run([self.losses, self.update_op], td_map)

        return losses
    
    def update_with_pd(self, obs, acs, advs, values, pds, lrmult, im_weight, pi_lr_mask=1.0):
        td_map = {self.ob: obs, self.ac: acs, self.atarg: advs, self.pd_ph: pds, self.im_weight: im_weight,
                  self.ret: values, self.lrmult: lrmult, self.pi_lr_mask: pi_lr_mask}
        losses, _ = self.sess.run([self.losses, self.update_op], td_map)

        return losses

    def get_pd(self, ob):
        return self.sess.run(self.pi.pd.flat, feed_dict={self.pi.ob: ob})