import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle
from gym import spaces
from pybullet_envs.deep_mimic.env.env import Env
from pybullet_envs.deep_mimic.env.action_space import ActionSpace
from pybullet_utils import bullet_client
from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger
import time

import motion_capture_data
import EnvBlock.leopard_stable_pd as leopard_stable_pd
import pybullet_data
import pybullet as p1
import random

def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if (arg_file != ''):
        path = pybullet_data.getDataPath() + "/args/" + arg_file
        succ = arg_parser.load_file(path)
        Logger.print2(arg_file)
        assert succ, Logger.print2('Failed to load args from: ' + arg_file)
    return arg_parser


# cmd = '--arg_file train_humanoid3d_run_args.txt'
cmd = '--arg_file train_leopard_walk_args.txt'
args = cmd.split()
update_timestep = 1. / 240.


class DMLeopardEnv(gym.Env):
    metadata = {'render.modes': ['leopard']}

    # test 的时候如果 enable_draw=False 会报错
    def __init__(self, pybullet_client=None, 
                        motion_file=None, seg_init_frames=None,
                        periodic=False, 
                        seg_start_frames=None, phase_instr=None,
                        seg_end_frames=None, tm_args=None):     # use tm_args to distinguish with args
        
        self.periodic = periodic
        self.motion_file = motion_file
        self.tm_args = tm_args

        if not hasattr(tm_args, 'phase_instr'):
            tm_args.phase_instr = 'normal'
            tm_args.view_rad = None
            tm_args.ds_step = None
        if not hasattr(tm_args, 'noise_scale'):
            tm_args.noise_scale = 0.0
        if not hasattr(tm_args, 'min_start_frame'):
            tm_args.min_start_frame = 20
        if not hasattr(tm_args, 'pos_diff'):
            tm_args.pos_diff = 1e6          # no constraint on pos_diff
        if not hasattr(tm_args, 'toe'):
            tm_args.toe = 1.0
        if not hasattr(tm_args, 'no_clip'):
            tm_args.no_clip = False
        if not hasattr(tm_args, 'time_shift'):
            tm_args.time_shift = None
        if not hasattr(tm_args, 'start_seg') or tm_args.start_seg is None:
            tm_args.start_seg = None
        if not hasattr(tm_args, 'end_seg') or tm_args.end_seg is None:
            tm_args.end_seg = None
            
        if not hasattr(tm_args, 'startFrame'):
            tm_args.startFrame = None
        if not hasattr(tm_args, 'endFrame'):
            tm_args.endFrame = None

        
        self.maxForces = [100] * 18

        self.enable_draw = tm_args.enable_draw
        if phase_instr is not None:
            self.phase_instr = phase_instr
        else:
            self.phase_instr = tm_args.phase_instr

        self.view_rad = tm_args.view_rad
        self.ds_step = tm_args.ds_step
        self.duration = tm_args.duration
        self.random_init = tm_args.random_init
        self.noise_scale = tm_args.noise_scale
        self.random_scale = tm_args.random_scale
        self.RL_weight = tm_args.rl_weight

        self.start_seg = self.tm_args.start_seg
        self.end_seg = self.tm_args.end_seg

        self.min_frame_start_time = tm_args.min_start_frame * self.duration        

        assert self.motion_file is not None, 'No motion file name'

        observation_dim = self.get_state_size()
        observation_high = np.array(
            [np.finfo(np.float32).max] * observation_dim)
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high)
        self.observation_dim = observation_dim

        action_low, action_high = self.build_action_bound_min(), self.build_action_bound_max()
        self.action_space = spaces.Box(
            np.array(action_low), np.array(action_high))
        self.action_dim = 12
        self._pybullet_client = pybullet_client
        self._isInitialized = False
        self._useStablePD = True
        self._arg_parser = build_arg_parser(args)  # arg_parser

        if seg_init_frames is None:
            self.seg_init_frames = seg_start_frames
        else:
            self.seg_init_frames = seg_init_frames
        self.seg_start_frames = seg_start_frames
        self.seg_end_frames = seg_end_frames

        # self.given_rn = None
        self.weighted_action = None
        self.reset()

        
    def reset(self):
        if not self._isInitialized:
            if self.enable_draw:
                self._pybullet_client = bullet_client.BulletClient(
                    connection_mode=p1.GUI)
            else:
                self._pybullet_client = bullet_client.BulletClient()

            self._pybullet_client.setAdditionalSearchPath(
                pybullet_data.getDataPath())
                
            self._planeId = self._pybullet_client.loadURDF(
                "plane_implicit.urdf")  # leopard

            self._pybullet_client.setGravity(0, 0, -9.8)

            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=10)
            self._pybullet_client.changeDynamics(
                self._planeId, linkIndex=-1, lateralFriction=0.9)

            self._mocapData = motion_capture_data.MotionCaptureData()

            motionPath = os.path.join(pybullet_data.getDataPath(), self.motion_file)
            self._mocapData.Load(motionPath, 0, self.tm_args.endFrame)
            self._mocapData.appendDuration2Frames(self.duration)

            timeStep = update_timestep
            useFixedBase = False
            self._leopard = leopard_stable_pd.LeopardStablePD(self._pybullet_client, self._mocapData,
                                                              timeStep, useFixedBase, self._arg_parser, phase_instr=self.phase_instr,
                                                              periodic=self.periodic, tm_args=self.tm_args)
            self._isInitialized = True

            self._pybullet_client.setTimeStep(timeStep)
            self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

            
            rnrange = 1000
            rn = random.randint(0, rnrange)   # 生成 (0, 1000) 的随机数
            self.t = float(rn) / rnrange / self.random_scale * self._leopard.getCycleTime()


        #print("self._leopard.getCycleTime()", self._leopard.getCycleTime())
        rnrange = 1000
        rn = random.randint(0, rnrange)   # 生成 (0, 1000) 的随机数   

        if self.start_seg is not None:
            total_time_shift = self.seg_start_frames[self.start_seg] * self.duration        
        else:
            total_time_shift = 0.0        

        if self.tm_args.time_shift is not None:
            total_time_shift += self.tm_args.time_shift 
            
        if 'Arb' in self.random_init:     # random init starting from a random sample
            if self.seg_end_frames is None:
                cycleTime = self._leopard.getCycleTime()
            else:
                cycleTime = self.duration * (self.seg_end_frames[self.end_seg] - self.seg_start_frames[self.start_seg])

            self.rn = rn
            # if self.given_rn is not None:
            #     rn = self.given_rn
            self.t = float(rn) / rnrange / self.random_scale * cycleTime  # 用于从一个循环的随机一点开始
        elif 'Seg' in self.random_init:
            # random init starting from some start point of segments
            start_point = np.random.randint(0, len(self.seg_init_frames))
            start_frames = self.seg_init_frames[start_point][0]
            end_frames = self.seg_init_frames[start_point][1]
            self.t = (start_frames + float(rn) / rnrange / self.random_scale * (end_frames - start_frames)) * self.duration
            # print('Start Partition is ', start_point)
        elif 'Fixed' in self.random_init:
            self.t = self.tm_args.start_time
            print('Start from ', self.t)
            # self.cur_start_point = 0
        else:           # False is default
            self.t = 0    # no random init
            self.cur_start_point = 0

        self.t += total_time_shift
            
        if self.t < self.min_frame_start_time:
            self.t = self.t + self.min_frame_start_time
        self.cur_start_point = self.t / 0.01667
        
        self._leopard.setSimTime(self.t)
        self._leopard.resetPose()  
        self.needs_update_time = self.t - 1  # force update

        state = self._leopard.getState()
        return self.state_wrapper(state)
        # return np.array(state)

    # make sure ob is np.array
    def state_wrapper(self, ob):
        return np.array(ob)

    def set_start_end_seg(self, start_seg, end_seg):
        self.start_seg = start_seg
        self.end_seg = end_seg
        # self._mocapData.reset_motion_data(self.seg_start_frames[start_seg], self.seg_end_frames[end_seg])

    def return_sim_pose(self):
        return self._leopard.return_sim_pose()

    def return_kin_pose(self):        
        return self._leopard.return_kin_pose() 

    def get_time(self):
        return self.t

    def getBasePosOrn(self):
        kinPos, kinOrn = self._leopard._pybullet_client.getBasePositionAndOrientation(
            self._leopard._kin_model)
        simPos, simOrn = self._leopard._pybullet_client.getBasePositionAndOrientation(
            self._leopard._sim_model)
        
        return np.array(kinPos), np.array(kinOrn), np.array(simPos), np.array(simOrn)

    def getLinkPosOrn(self, index=0):
        kinPos, kinOrn = self._leopard._pybullet_client.getLinkState(self._leopard._kin_model, index)[:2]
        simPos, simOrn = self._leopard._pybullet_client.getLinkState(self._leopard._sim_model, index)[:2]
        
        return np.array(kinPos), np.array(kinOrn), np.array(simPos), np.array(simOrn)
        
    def reset_given_time(self, t):        
        self._leopard.setSimTime(self.t)
        self._leopard.resetPose() 

    def get_leopard(self):
        return self._leopard

    def get_mocap_data(self):
        return self._mocapData

    def get_frame(self):
        return self._leopard._frame + self._leopard._frameFraction
    
    def refine_phase(self, index, start_index, total_frame):
        return (self._leopard._frame + self._leopard._frameFraction + start_index) / total_frame

    def get_time_info(self, info, last_time):
        print(info, time.time() - last_time)
        return time.time()

    def step(self, action):
        action = action.copy()
        info = self.set_action(action)
        need_update = True
        while need_update:
            self.update(update_timestep)
            done = self.is_episode_end()
            if done:
                obs = self._leopard.getState()
                reward = self.calc_reward()
                
                info.update({'stateVector': self.state_wrapper(self._leopard.stateVector)})
                return self.state_wrapper(obs), reward, True, info
            else:
                need_update = not self.need_new_action()
                
        # pos, ort, _, __ = self._leopard.getBasePosAndOrt()
        # euler = self._pybullet_client.getEulerFromQuaternion(ort)
        # print(ort, euler)

        obs = self._leopard.getState()
        reward = self.calc_reward()

        info.update({'stateVector': self.state_wrapper(self._leopard.stateVector)})
        return self.state_wrapper(obs), reward, False, info

    def render(self, mode='leopard', close=False):
        # a = 1
        leopardPos, leopardOrn = self._leopard._pybullet_client.getBasePositionAndOrientation(
            self._leopard._kin_model)    # _sim_model
        # print("leopardPos:",leopardPos)
        # print("leopardOrn:",leopardOrn)
        if (close == False):

            camInfo = self._leopard._pybullet_client.getDebugVisualizerCamera()
            curTargetPos = camInfo[11]
            # self._leopard._pybullet_client.resetDebugVisualizerCamera(3, -90 * 0, -15, leopardPos)  # (3, -90 * 0, -15)->(,,俯仰角)
            self._leopard._pybullet_client.resetDebugVisualizerCamera(
                1, 45, -10, leopardPos)

    def convert_pose_to_action(self, pose):
        # turn pose (quaternion representation) into
        # action (axis angle representation used as output from policy network)
        return pose[7:7+12]

    def need_new_action(self):
        if self.t >= self.needs_update_time:
            self.needs_update_time = self.t + 1. / 30  # 1 / 30   PD控制的周期
            return True
        return False

    def get_kin_action(self):
        kinPose = self._leopard.computePose(self._leopard._frameFraction)
        return self.convert_pose_to_action(kinPose)

    def set_action(self, action):
        action = self.clip_fun(action)

        kinPose = self._leopard.computePose(self._leopard._frameFraction)
        kin_action = self.convert_pose_to_action(kinPose)        

        w_RL = self.RL_weight
        w_motion = 1 - w_RL
        weighted_action = [(w_RL * action[i] + w_motion * kin_action[i])
                        for i in range(len(action))]

        self.desiredPose = self._leopard.convertActionToPose(weighted_action)

        return {
            'kin_action': kin_action, 
            'weighted_action': weighted_action,
            'clipped_action': action,
        }


    def step_without_rm(self, action):
        action = action.copy()
        info = self.set_action_without_rm(action)
        need_update = True
        while need_update:
            self.update(update_timestep)
            done = self.is_episode_end()
            if done:
                obs = self._leopard.getState()
                reward = self.calc_reward()
                # return np.array(obs), reward, True, {}

                # # if self.t is large enough, then do not use "continue initialization"
                # if self.t <= 0.9 * self._mocapData.NumFrames() * self.duration:
                #     self.reset_time = self.t
                # else:
                #     self.reset_time = None
                info.update({'stateVector': self.state_wrapper(self._leopard.stateVector)})
                return self.state_wrapper(obs), reward, True, info
            else:
                need_update = not self.need_new_action()
                
        # pos, ort, _, __ = self._leopard.getBasePosAndOrt()
        # euler = self._pybullet_client.getEulerFromQuaternion(ort)
        # print(ort, euler)

        obs = self._leopard.getState()
        reward = self.calc_reward()

        info.update({'stateVector': self.state_wrapper(self._leopard.stateVector)})
        return self.state_wrapper(obs), reward, False, info

    def set_action_without_rm(self, action):
        sim_action = self.clip_fun(action.copy())

        kinPose = self._leopard.computePose(self._leopard._frameFraction)
        kin_action = self.convert_pose_to_action(kinPose)        

        w_RL = self.RL_weight
        w_motion = 1 - w_RL
        weighted_action = [(w_RL * sim_action[i] + w_motion * kin_action[i])
                        for i in range(len(sim_action))]

        self.desiredPose = self._leopard.convertActionToPose(action)

        return {
            'kin_action': kin_action, 
            'weighted_action': weighted_action,
            'clipped_action': action,
        }


    def update(self, timeStep):
        # print("pybullet_deep_mimic_env:update timeStep=",timeStep," t=",self.t)
        self._pybullet_client.setTimeStep(timeStep)
        self._leopard._timeStep = timeStep

        self.t += timeStep
        # print("================self.t in file dmLeopard.py line 254:==================", self.t)
        self._leopard.setSimTime(self.t)

        if self.desiredPose:
            kinPose = self._leopard.computePose(self._leopard._frameFraction)
            
            self._leopard.initializePose(self._leopard._poseInterpolator,
                                            self._leopard._kin_model,
                                            initBase=True)
            if self._useStablePD:
                # usePythonStablePD = False
                usePythonStablePD = True
                if usePythonStablePD:
                    # print("self.desiredPose in file dmLeopard.py ", len(self.desiredPose), self.desiredPose)
                    taus = self._leopard.computePDForces(self.desiredPose,
                                                            desiredVelocities=None,
                                                            maxForces=self.maxForces)
                    self._leopard.applyPDForces(taus)
                else:
                    self._leopard.computeAndApplyPDForces(self.desiredPose,
                                                            maxForces=self.maxForces)
            else:
                self._leopard.setJointMotors(
                    self.desiredPose, maxForces=self.maxForces)

            self._pybullet_client.stepSimulation()

    def calc_reward(self):
        reward = self._leopard.getReward()
        return reward

    def is_episode_end(self):
        isEnded = self._leopard.terminates()
        # also check maximum time, 20 seconds (todo get from file)
        # print("self.t=",self.t)
        if (self.t > 50):
            # print(self.t)
            isEnded = True

        return isEnded
            
    def get_state_size(self):
        dim = 156
        if self.phase_instr == 'normal':
            dim = dim + 2
        elif self.tm_args.points:
            dim += 3 * 2 * self.view_rad        # base position
            if self.tm_args.velocity:
                dim += 2 * 2 * self.view_rad    # x, y velocity
            if self.tm_args.use_ort:
                dim += 4 * 2 * self.view_rad    # quanterion info
        else:
            raise NotImplementedError

        print('State Size is ', dim)
        return dim

    def get_action_size(self):
        ctrl_size = 19  # numDof
        root_size = 7
        return ctrl_size - root_size

    def build_action_bound_min(self):

        out_scale = [
            -0, -0, -0,
            -0, -0, -0,
            -0, -0, -0,
            -0, -0, -0
        ]

        return out_scale

    def build_action_bound_max(self):

        out_scale = [
            -0, -0, -0,
            -0, -0, -0,
            -0, -0, -0,
            -0, -0, -0
        ]

        return out_scale

    def clip_fun(self, nums):
        """
        对列表中的每个元素按照不同的上下限进行clip
        :param list: 1D list
        :return: 1D list
        """

        min_bound = [-0.10, -3.00, -3.00,
                     -0.10, -3.00, -3.00,
                     -0.10, -3.00, -3.00,
                     -0.10, -3.00, -3.00]

        max_bound = [0.10, 3.00, 3.00,
                     0.10, 3.00, 3.00,
                     0.10, 3.00, 3.00,
                     0.10, 3.00, 3.00]

        for i in range(0, 12):
            nums[i] = np.clip(nums[i], min_bound[i], max_bound[i])
        return nums


    def clip_sum_fun(self, nums):
        """
        对列表中的每个元素按照不同的上下限进行clip
        :param list: 1D list
        :return: 1D list
        """

        min_bound = [-1.10, -3.00, -3.00,
                     -1.10, -3.00, -3.00,
                     -1.10, -3.00, -3.00,
                     -1.10, -3.00, -3.00]

        max_bound = [1.10, 3.00, 3.00,
                     1.10, 3.00, 3.00,
                     1.10, 3.00, 3.00,
                     1.10, 3.00, 3.00]

        for i in range(0, 12):
            nums[i] = np.clip(nums[i], min_bound[i], max_bound[i])
        return nums
