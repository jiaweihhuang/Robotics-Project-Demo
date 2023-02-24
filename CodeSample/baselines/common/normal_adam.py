import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
from mpi4py import MPI


class NormalAdam(object):
    def __init__(self, var_list, *, rms_var_list, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float64')
        self.v = np.zeros(size, 'float64')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)

        '''can not set together because rms is tf float64 '''
        self.rms_var_list = rms_var_list
        self.rms_setfromflat = U.SetFromFlat(rms_var_list, dtype=tf.float64)
        self.rms_getflat = U.GetFlat(rms_var_list)

        self.rank = MPI.COMM_WORLD.Get_rank()

    def update(self, globalg, stepsize):
        self.t += 1
        globalg = globalg.astype(np.float64)
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)
    