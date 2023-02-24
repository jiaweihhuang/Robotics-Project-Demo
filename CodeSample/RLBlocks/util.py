import RLBlocks.mlp_policy as mlp_policy 
import os
import zipfile
import cloudpickle
import tempfile
import tensorflow as tf
import numpy as np



def load_state(fname, sess):
    saver = tf.train.Saver()
    saver.restore(sess, fname)


def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)


class ActWrapper(object):
    def __init__(self, pi):
        self._pi = pi
        self._act = self._pi._act

    @staticmethod
    def load(path, ob_shape, ac_space, sess, hid_size=128, num_hid_layers=2, 
                    activation='tanh', name='pi', ob=None, trainable=True):
        with open(path, "rb") as f:
            model_data = cloudpickle.load(f)
        pi = mlp_policy.MlpPolicy(name=name, ob_shape=ob_shape, ac_space=ac_space, trainable=trainable,
                                hid_size=hid_size, num_hid_layers=num_hid_layers, activation=activation, ob=ob)
        act = pi._act
        
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_state(os.path.join(td, "model"), sess)

        return ActWrapper(pi)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def get_act_pd_with_ob(self, ob):
        return self._pi.get_pd(ob)

    def get_act_pd(self,):
        return self._pi.pd.flat

    def get_pi(self):
        return self._pi

    def save(self, logdir, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logdir, "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(
                                file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data), f)






class ArgParser(object):
    global_parser = None

    def __init__(self):
        self._table = dict()
        return

    def clear(self):
        self._table.clear()
        return

    def load_args(self, arg_strs):
        succ = True
        vals = []
        curr_key = ''

        for str in arg_strs:
            if not (self._is_comment(str)):
                is_key = self._is_key(str)
                if (is_key):
                    if (curr_key != ''):
                        if (curr_key not in self._table):
                            self._table[curr_key] = vals

                    vals = []
                    curr_key = str[2::]
                else:
                    vals.append(str)

        if (curr_key != ''):
            if (curr_key not in self._table):
                self._table[curr_key] = vals

            vals = []

        return succ

    def load_file(self, filename):
        succ = False
        with open(filename, 'r') as file:
            lines = RE.split(r'[\n\r]+', file.read())
            file.close()

            arg_strs = []
            for line in lines:
                if (len(line) > 0 and not self._is_comment(line)):
                    arg_strs += line.split()

            succ = self.load_args(arg_strs)
        return succ

    def has_key(self, key):
        return key in self._table

    def parse_string(self, key, default=''):
        str = default
        if self.has_key(key):
            str = self._table[key][0]
        return str

    def parse_strings(self, key, default=[]):
        arr = default
        if self.has_key(key):
            arr = self._table[key]
        return arr

    def parse_int(self, key, default=0):
        val = default
        if self.has_key(key):
            val = int(self._table[key][0])
        return val

    def parse_ints(self, key, default=[]):
        arr = default
        if self.has_key(key):
            arr = [int(str) for str in self._table[key]]
        return arr

    def parse_float(self, key, default=0.0):
        val = default
        if self.has_key(key):
            val = float(self._table[key][0])
        return val

    def parse_floats(self, key, default=[]):
        arr = default
        if self.has_key(key):
            arr = [float(str) for str in self._table[key]]
        return arr

    def parse_bool(self, key, default=False):
        val = default
        if self.has_key(key):
            val = self._parse_bool(self._table[key][0])
        return val

    def parse_bools(self, key, default=[]):
        arr = default
        if self.has_key(key):
            arr = [self._parse_bool(str) for str in self._table[key]]
        return arr

    def _is_comment(self, str):
        is_comment = False
        if (len(str) > 0):
            is_comment = str[0] == '#'

        return is_comment
        
    def _is_key(self, str):
        is_key = False
        if (len(str) >= 3):
            is_key = str[0] == '-' and str[1] == '-'

        return is_key

    def _parse_bool(self, str):
        val = False
        if (str == 'true' or str == 'True' or str == '1' 
            or str == 'T' or str == 't'):
            val = True
        return val
