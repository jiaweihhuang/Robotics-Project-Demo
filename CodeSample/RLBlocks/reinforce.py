import random
import numpy as np
import tensorflow as tf
tfd = tf.contrib.distributions


def normalize(data, mean=0.0, std=1.0):
    n_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
    return n_data * (std + 1e-8) + mean


def build_mlp(input_placeholder, output_size, scope,
              n_layers=2, size=64, activation=tf.tanh,
              output_activation=None):

    with tf.variable_scope(scope):
        # YOUR_CODE_HERE
        out = input_placeholder
        for l in range(n_layers):
            out = tf.layers.dense(inputs=out, units=size,
                                  activation=activation)
        out = tf.layers.dense(inputs=out, units=output_size,
                              activation=output_activation)
        return out


class PolicyGradient(object):
    def __init__(self, ob_dim, ac_dim, discrete,
                 gamma=1.0, gae_lambda=1.0, learning_rate=5e-3,
                 reward_to_go=True, normalize_advantages=True, nn_baseline=False,
                 n_layers=1, size=32):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate
        self.reward_to_go = reward_to_go
        self.normalize_advantages = normalize_advantages
        self.nn_baseline = nn_baseline
        self.n_layers = n_layers
        self.size = size
        self.create_variables()
        self.paths = []
        #
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()  # equivalent to `with sess:`
        tf.global_variables_initializer().run()  # pylint: disable=E1101

    def sample_action(self, states):
        ac = self.sess.run(self.sy_sampled_nac, feed_dict={
                           self.sy_ob_no: [states]})
        ac = ac[0]
        return ac

    def create_variables(self):
        self.sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)
        # Observations are input for everything: sampling actions, baselines, policy gradients
        self.sy_ob_no = tf.placeholder(
            shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        # Actions are input when computing policy gradient updates
        if self.discrete:
            self.sy_nac = tf.placeholder(
                shape=[None], name="ac", dtype=tf.int32)
        else:
            self.sy_nac = tf.placeholder(
                shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        # Advantages are input when computing policy gradient updates
        self.sy_adv_n = tf.placeholder(
            shape=[None], name="adv", dtype=tf.float32)
        # Networks
        if self.discrete:
            # YOUR_CODE_HERE
            # Compute stochastic policy over discrete actions
            self.sy_logits_na = build_mlp(self.sy_ob_no, self.ac_dim, "policy",
                                          n_layers=self.n_layers, size=self.size)
            self.sy_logp_na = tf.nn.log_softmax(
                self.sy_logits_na)  # logprobability of actions
            # Sample an action from the stochastic policy
            self.sy_sampled_nac = tf.multinomial(self.sy_logits_na, 1)
            self.sy_sampled_nac = tf.reshape(self.sy_sampled_nac, [-1])
            # Likelihood of chosen action
            self.sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.sy_nac, logits=self.sy_logits_na)
            # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
            self.sy_oldlogits_na = tf.placeholder(
                shape=[None, self.ac_dim], name='oldlogits', dtype=tf.float32)
            self.sy_oldlogp_na = tf.nn.log_softmax(
                self.sy_oldlogits_na)  # log(p(a))
            self.sy_oldp_na = tf.exp(self.sy_oldlogp_na)  # p(a)
            sy_n = tf.shape(self.sy_ob_no)[0]
            self.sy_kl = tf.reduce_sum(
                self.sy_oldp_na * (self.sy_oldlogp_na - self.sy_logp_na)) / tf.to_float(sy_n)
            self.sy_p_na = tf.exp(self.sy_logp_na)
            self.sy_ent = tf.reduce_sum(- self.sy_p_na *
                                        self.sy_logp_na) / tf.to_float(sy_n)
        else:
            # YOUR_CODE_HERE
            # Compute Gaussian stochastic policy over continuous actions.
            # The mean is a function of observations, while the variance is not.
            self.sy_mean_na = build_mlp(
                self.sy_ob_no, self.ac_dim, "policy", n_layers=self.n_layers, size=self.size)
            self.sy_logstd = tf.Variable(
                tf.zeros([1, self.ac_dim]), name="policy/logstd", dtype=tf.float32)
            self.sy_std = tf.exp(self.sy_logstd)
            # action distribution before update
            self.sy_ac_dist = tfd.MultivariateNormalDiag(
                loc=self.sy_mean_na, scale_diag=self.sy_std)
            # mean and variance BEFORE update (only used for KL diagnostics)
            self.sy_oldmean = tf.placeholder(
                shape=[None, self.ac_dim], name='oldmean', dtype=tf.float32)
            self.sy_oldlogstd = tf.placeholder(
                shape=[1, self.ac_dim], name='oldlogstd', dtype=tf.float32)
            self.sy_oldstd = tf.exp(self.sy_oldlogstd)
            self.sy_oldac_dist = tfd.MultivariateNormalDiag(
                loc=self.sy_oldmean, scale_diag=self.sy_oldstd)
            # Sample an action from the stochastic policy
            self.sy_sampled_z = tf.random_normal(tf.shape(self.sy_mean_na))
            self.sy_sampled_nac = self.sy_mean_na + self.sy_std * self.sy_sampled_z
            # Likelihood of chosen action
            self.sy_z = (self.sy_nac - self.sy_mean_na) / self.sy_std
            self.sy_logprob_n = -0.5 * \
                tf.reduce_sum(tf.square(self.sy_z), axis=1)
            #
            self.sy_kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(
                self.sy_ac_dist, self.sy_oldac_dist))
            self.sy_ent = tf.reduce_mean(self.sy_ac_dist.entropy())
        # Loss Function and Training Operation
        # Loss function that we'll differentiate to get the policy gradient.
        # Note: no gradient will flow through sy_adv_n, because it's a placeholder.
        self.loss = -tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n)

        self.update_op = tf.train.AdamOptimizer(
            self.sy_stepsize).minimize(self.loss)
        # Optional Baseline
        if self.nn_baseline:
            self.baseline_prediction = tf.squeeze(build_mlp(self.sy_ob_no, 1, "nn_baseline",
                                                            n_layers=self.n_layers, size=self.size))
            # Define placeholders for targets, a loss function and an update op for fitting a
            # neural network baseline. These will be used to fit the neural network baseline.
            # YOUR_CODE_HERE
            self.sy_target_n = tf.placeholder(
                shape=[None], name="target", dtype=tf.float32)
            self.baseline_loss = tf.nn.l2_loss(
                self.baseline_prediction - self.sy_target_n)
            self.baseline_update_op = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.baseline_loss)

    def store_rollout(self, paths):
        self.paths = paths

    def update_model(self, seg, stepsize):
        # np.concatenate([path["observation"] for path in self.paths])
        ob_no = seg["ob"]
        # ob_no = (ob_no - np.mean(ob_no,axis=0))/(np.std(ob_no,axis=0) + 1e-8)
        # np.concatenate([path["action"] for path in self.paths])
        ac_nac = seg["ac"]
        #
        if self.discrete:
            old_logit_na = self.sess.run(
                self.sy_logits_na, feed_dict={self.sy_ob_no: ob_no})
        else:
            old_mean = self.sess.run(self.sy_mean_na, feed_dict={
                                     self.sy_ob_no: ob_no})
            # , feed_dict={self.sy_ob_no: ob_no})
            oldlogstd = self.sess.run(self.sy_logstd)
        # Computing Q-values
        # YOUR_CODE_HERE
        q_n = []
        q = 0
        q_path = []
        # Dynamic programming over reversed path
        for rew in reversed(seg["rew"]):
            q = rew + self.gamma * q
            q_path.append(q)
        q_path.reverse()
        # Append these q values
        if not self.reward_to_go:
            q_path = [q_path[0]] * len(q_path)
        q_n.extend(q_path)

        # Computing Baselines
        if self.nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            b_n = self.sess.run(self.baseline_prediction,
                                feed_dict={self.sy_ob_no: ob_no})
            b_n = normalize(b_n, np.mean(q_n), np.std(q_n))
            # Generalized advantage estimation
            adv_n = []
            idx = 0
            adv = 0
            adv_path = []
            V_next = 0
            idx += len(seg["rew"])
            # Dynamic programming over reversed path
            for rew, V in zip(reversed(seg["rew"]), b_n[idx - 1:None:-1]):
                bellman_error = rew + self.gamma * V_next - V
                adv = bellman_error + self.gae_lambda * self.gamma * adv
                adv_path.append(adv)
                V_next = V
            adv_path.reverse()
            # Append these advantage values
            if not self.reward_to_go:
                adv_path = [adv_path[0]] * len(adv_path)
            adv_n.extend(adv_path)

            # Compute a GAE version of q_n to use when fitting the baseline
            q_n = b_n + adv_n
        else:
            adv_n = q_n.copy()
        adv_n = np.array(adv_n)
        # Advantage Normalization
        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            # YOUR_CODE_HERE
            adv_n = normalize(adv_n)
        # Optimizing Neural Network Baseline
        if self.nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            # YOUR_CODE_HERE
            q_normalized_n = normalize(q_n)
            self.sess.run(self.baseline_update_op,
                          feed_dict={self.sy_ob_no: ob_no, self.sy_target_n: q_normalized_n})
        # Performing the Policy Update
        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        # YOUR_CODE_HERE
        self.sess.run(self.update_op,
                      feed_dict={self.sy_ob_no: ob_no, self.sy_nac: ac_nac, self.sy_adv_n: adv_n,
                                 self.sy_stepsize: stepsize})
        #
        if self.discrete:
            kl = self.sess.run(self.sy_kl, feed_dict={
                               self.sy_ob_no: ob_no, self.sy_oldlogits_na: old_logit_na})
            ent = self.sess.run(self.sy_ent, feed_dict={self.sy_ob_no: ob_no})
        else:
            kl = self.sess.run(self.sy_kl, feed_dict={
                               self.sy_ob_no: ob_no, self.sy_oldmean: old_mean, self.sy_oldlogstd: oldlogstd})
            ent = self.sess.run(self.sy_ent, feed_dict={self.sy_ob_no: ob_no})
        return kl, ent
