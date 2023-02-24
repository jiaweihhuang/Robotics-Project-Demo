import numpy as np
import pickle
import random
import tensorflow as tf

class RB_BasicClass(object):    
    def size(self):
        return len(self.obs_buffer)

    @staticmethod
    def dump(path, rb):
        with open(path, 'wb') as f:
            pickle.dump(rb, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class ReplayBuffer(RB_BasicClass):
    def __init__(self, buffer_size, obs_dim, act_dim, use_act=False):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.obs_buffer = np.zeros(shape=[self.buffer_size, obs_dim])
        if use_act:
            self.target_buffer = np.zeros(shape=[self.buffer_size, act_dim])
        else:
            self.target_buffer = np.zeros(shape=[self.buffer_size, 2 * act_dim])     # mean & logstd => 2 * act_dim
        self.count = 0

    def add(self, obs, index, target):
        self.obs_buffer[self.count % self.buffer_size] = obs
        self.target_buffer[self.count % self.buffer_size] = target
        self.count += 1

    def sample(self, batch_size):
        if self.count < self.buffer_size:
            batch_indices = np.random.randint(low=0, high=self.count, size=batch_size)
        else:
            batch_indices = np.random.randint(low=0, high=self.buffer_size, size=batch_size)
        
        obs = np.take(self.obs_buffer, batch_indices, axis=0)
        targets = np.take(self.target_buffer, batch_indices, axis=0)

        # return two np.array with shape [batch_size, obs_dim] and [batch_size, 1]
        return obs, None, targets


class TF_ReplayBuffer(RB_BasicClass):
    @staticmethod
    def dump(path, obs, targets):
        data = {
            'obs': obs,
            'targets': targets,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path, batch_size):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return TF_ReplayBuffer(data['obs'], data['targets'], batch_size)

    def size(self):
        return self.buffer_size

    def __init__(self, *args, **kwargs):
        self._init(*args, **kwargs)
        
    def __init__(self, obs_inital_value, targets_initial_value, batch_size):
        self.buffer_size = obs_inital_value.shape[0]
        self.obs_dim = obs_inital_value.shape[1]
        self.targets_dim = targets_initial_value.shape[1]

        self.obs_buffer = tf.Variable(tf.constant(value=obs_inital_value, dtype=tf.float32), trainable=False)
        self.targets_buffer = tf.Variable(tf.constant(value=targets_initial_value, dtype=tf.float32), trainable=False)

        self.obs_ph = tf.placeholder(shape=[None, self.obs_dim], dtype=tf.float32)
        self.targets_ph = tf.placeholder(shape=[None, self.targets_dim], dtype=tf.float32)
        self.start_index = tf.placeholder(shape=[], dtype=tf.int32)
        self.slice_size = tf.placeholder(shape=[], dtype=tf.int32)

        self.assign_list = [
            tf.assign(self.obs_buffer[self.start_index:self.start_index+self.slice_size, :], self.obs_ph),
            tf.assign(self.targets_buffer[self.start_index:self.start_index+self.slice_size, :], self.targets_ph),
        ]

        ''' Build Uniform Sampler '''
        self.batch_size = batch_size
        self.uniform_index = tf.random.uniform(shape=[self.batch_size], minval=0, maxval=self.buffer_size, dtype=tf.int32)

        self.obs_samples = tf.gather(self.obs_buffer, self.uniform_index)
        self.targets_samples = tf.gather(self.targets_buffer, self.uniform_index)

        self.index = 0

        tf.get_default_session().run(
            tf.initialize_variables([self.obs_buffer, self.targets_buffer])
        )

    def set_values(self, obs, targets, start_index, slice_size):
        tf.get_default_session().run(
            self.assign_list,
            feed_dict = {
                self.obs_ph: obs,
                self.targets_ph: targets,
                self.start_index: start_index,
                self.slice_size: slice_size,
            }
        )

    def update(self, obs, targets):
        new_data_size = obs.shape[0]
        if self.index + new_data_size > self.buffer_size:
            assert self.index + new_data_size < 2 * self.buffer_size
            threshold = self.buffer_size - self.index
            self.set_values(obs[:threshold, :], targets[:threshold, :], start_index=self.index, slice_size=threshold)
            self.set_values(obs[threshold:, :], targets[threshold:, :], start_index=0, slice_size=new_data_size - threshold)
        else:
            self.set_values(obs, targets, start_index=self.index, slice_size=new_data_size)
        
        self.index = (self.index + new_data_size) % self.buffer_size
        print('Update Data Size is ', new_data_size)
        print('Current start index is ', self.index)


class TF_ReplayBuffer_with_Indices(RB_BasicClass):
    @staticmethod
    def dump(path, obs, targets, indices):
        data = {
            'obs': obs,
            'targets': targets,
            'indices': indices,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path, batch_size):        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return TF_ReplayBuffer_with_Indices(data['obs'], data['targets'], data['indices'], batch_size)

    def size(self):
        return self.buffer_size

    def __init__(self, *args, **kwargs):
        self._init(*args, **kwargs)
        
    def _init(self, obs_inital_value, targets_initial_value, indices_initial_value, batch_size):
        self.buffer_size = obs_inital_value.shape[0]
        self.obs_dim = obs_inital_value.shape[1]
        self.targets_dim = targets_initial_value.shape[1]

        self.obs_buffer = tf.Variable(tf.constant(value=obs_inital_value, dtype=tf.float32), trainable=False)
        self.targets_buffer = tf.Variable(tf.constant(value=targets_initial_value, dtype=tf.float32), trainable=False)
        print(indices_initial_value.shape)
        self.indices_buffer = tf.Variable(tf.constant(value=indices_initial_value, dtype=tf.float32, shape=indices_initial_value.shape), trainable=False)

        self.obs_ph = tf.placeholder(shape=[None, self.obs_dim], dtype=tf.float32)
        self.targets_ph = tf.placeholder(shape=[None, self.targets_dim], dtype=tf.float32)
        self.indices_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.start_index = tf.placeholder(shape=[], dtype=tf.int32)
        self.slice_size = tf.placeholder(shape=[], dtype=tf.int32)

        self.assign_list = [
            tf.assign(self.obs_buffer[self.start_index:self.start_index+self.slice_size, :], self.obs_ph),
            tf.assign(self.targets_buffer[self.start_index:self.start_index+self.slice_size, :], self.targets_ph),
            tf.assign(self.indices_buffer[self.start_index:self.start_index+self.slice_size, :], self.indices_ph)
        ]

        ''' Build Uniform Sampler '''
        self.batch_size = batch_size
        self.uniform_index = tf.random.uniform(shape=[self.batch_size], minval=0, maxval=self.buffer_size, dtype=tf.int32)

        self.obs_samples = tf.gather(self.obs_buffer, self.uniform_index)
        self.targets_samples = tf.gather(self.targets_buffer, self.uniform_index)
        self.indices_samples = tf.gather(self.indices_buffer, self.uniform_index)

        self.index = 0

        tf.get_default_session().run(
            tf.initialize_variables([self.obs_buffer, self.targets_buffer, self.indices_buffer])
        )

    def set_values(self, obs, targets, indices, start_index, slice_size):
        tf.get_default_session().run(
            self.assign_list,
            feed_dict = {
                self.obs_ph: obs,
                self.targets_ph: targets,
                self.indices_ph: indices,
                self.start_index: start_index,
                self.slice_size: slice_size,
            }
        )

    def update(self, obs, targets, indices):
        new_data_size = obs.shape[0]
        if self.index + new_data_size > self.buffer_size:
            assert self.index + new_data_size < 2 * self.buffer_size
            threshold = self.buffer_size - self.index
            self.set_values(obs[:threshold, :], targets[:threshold, :], indices[:threshold, :], start_index=self.index, slice_size=threshold)
            self.set_values(obs[threshold:, :], targets[threshold:, :], indices[threshold:, :], start_index=0, slice_size=new_data_size - threshold)
        else:
            self.set_values(obs, targets, indices, start_index=self.index, slice_size=new_data_size)
        
        self.index = (self.index + new_data_size) % self.buffer_size
        print('Update Data Size is ', new_data_size)
        print('Current start index is ', self.index)
        


class StateBuffer(RB_BasicClass):
    def __init__(self):
        self.obs_buffer = []

    def add(self, ob):
        self.obs_buffer.append(ob)

    def sample(self, num_samples):
        return random.choice(self.obs_buffer, num_samples)


class InitStateBuffer(RB_BasicClass):
    def __init__(self):
        self.obs_buffer = []

    def add(self, pose):
        self.obs_buffer.append(pose)

    def sample(self):
        return random.choice(self.obs_buffer)


class InitStateBuffer2():
    def __init__(self, max_size):
        self.obs_buffer = []
        self.counter = 0
        self.max_size = max_size

    def add(self, pose):
        if self.counter < self.max_size:
            self.obs_buffer.append(pose)
            self.counter += 1
        else:
            self.counter += 1
            self.obs_buffer[self.counter % self.max_size] = pose


    def sample(self):
        assert self.counter > 0, 'There is no data in buffer!!!'
        return random.choice(self.obs_buffer)
