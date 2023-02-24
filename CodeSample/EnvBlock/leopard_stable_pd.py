import EnvBlock.pd_controller_stable as pd_controller_stable

import math
import numpy as np
import time

# from pybullet_envs.deep_mimic.env 
import EnvBlock.leopardPoseInterpolator as leopardPoseInterpolator


FR_hip_motor = 1  # 前右 髋关节
FR_upper_leg = 2  # 前右 大腿
FR_lower_leg = 3  # 前右 小腿

FL_hip_motor = 4  # 前左 髋关节
FL_upper_leg = 5  # 前左 大腿
FL_lower_leg = 6  # 前左 小腿

RR_hip_motor = 7
RR_upper_leg = 8
RR_lower_leg = 9

RL_hip_motor = 10
RL_upper_leg = 11
RL_lower_leg = 12

joint_indices = [
            FR_hip_motor, FR_upper_leg, FR_lower_leg,
            FL_hip_motor, FL_upper_leg, FL_lower_leg,
            RR_hip_motor, RR_upper_leg, RR_lower_leg,
            RL_hip_motor, RL_upper_leg, RL_lower_leg
        ]

jointFrictionForce = 0

class PosePackage(object):
    def __init__(self, pose, noise_scale):
        self.pose = pose
        self.noise_scale = noise_scale
        self.random_wrapper = self.random_wrapper_rel

    def random_wrapper_rel(self, value):
        if type(value) is float:
            return value + value * self.noise_scale * np.random.randn()
        return [value[i] + value[i] * self.noise_scale * np.random.randn() for i in range(len(value))]

    def update(self):
        pose = self.pose
        self._basePos, self._baseOrn = self.random_wrapper(self.pose._basePos), self.random_wrapper(self.pose._baseOrn)
        self._baseLinVel, self._baseAngVel = self.random_wrapper(self.pose._baseLinVel), self.random_wrapper(self.pose._baseAngVel)
    
        self._FR_hip_motorRot = self.random_wrapper(pose._FR_hip_motorRot)
        self._FR_upper_legRot = self.random_wrapper(pose._FR_upper_legRot)
        self._FR_lower_legRot = self.random_wrapper(pose._FR_lower_legRot)
        self._FL_hip_motorRot = self.random_wrapper(pose._FL_hip_motorRot)
        self._FL_upper_legRot = self.random_wrapper(pose._FL_upper_legRot)
        self._FL_lower_legRot = self.random_wrapper(pose._FL_lower_legRot)
        self._RR_hip_motorRot = self.random_wrapper(pose._RR_hip_motorRot)
        self._RR_upper_legRot = self.random_wrapper(pose._RR_upper_legRot)
        self._RR_lower_legRot = self.random_wrapper(pose._RR_lower_legRot)
        self._RL_hip_motorRot = self.random_wrapper(pose._RL_hip_motorRot)
        self._RL_upper_legRot = self.random_wrapper(pose._RL_upper_legRot)
        self._RL_lower_legRot = self.random_wrapper(pose._RL_lower_legRot)

        self._FR_hip_motorVel = self.random_wrapper(pose._FR_hip_motorVel)
        self._FR_upper_legVel = self.random_wrapper(pose._FR_upper_legVel)
        self._FR_lower_legVel = self.random_wrapper(pose._FR_lower_legVel)
        self._FL_hip_motorVel = self.random_wrapper(pose._FL_hip_motorVel)
        self._FL_upper_legVel = self.random_wrapper(pose._FL_upper_legVel)
        self._FL_lower_legVel = self.random_wrapper(pose._FL_lower_legVel)
        self._RR_hip_motorVel = self.random_wrapper(pose._RR_hip_motorVel)
        self._RR_upper_legVel = self.random_wrapper(pose._RR_upper_legVel)
        self._RR_lower_legVel = self.random_wrapper(pose._RR_lower_legVel)
        self._RL_hip_motorVel = self.random_wrapper(pose._RL_hip_motorVel)
        self._RL_upper_legVel = self.random_wrapper(pose._RL_upper_legVel)
        self._RL_lower_legVel = self.random_wrapper(pose._RL_lower_legVel)



class LeopardStablePD(object):

    def __init__(self, pybullet_client, mocap_data, timeStep, 
                 useFixedBase=True, arg_parser=None, periodic=False, use_matrix=False,
                 tm_args=None, phase_instr=None):
        self._pybullet_client = pybullet_client
        self._mocap_data = mocap_data
        self._arg_parser = arg_parser
        self.tm_args = tm_args
        self.periodic = periodic
        self.phase_instr = phase_instr
        self.use_matrix = use_matrix

        self.ds_step = tm_args.ds_step
        self.view_rad = tm_args.view_rad
        self.noise_scale = tm_args.noise_scale

        self.stateVector = None

        print("=============================LOADING laikago!===============================")
        # flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER+self._pybullet_client.URDF_USE_SELF_COLLISION+self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        # self._sim_model = self._pybullet_client.loadURDF(
        #     "laikago/laikago.urdf", [0, 0.889540259, 0],
        #     globalScaling=1.0,
        #     useFixedBase=useFixedBase,
        #     flags=flags)
        # startPos = [0.007058990464444105, 0.03149299192130908, 0.4918981912395484]                                 # Laikago
        # startOrn = [0.005934649695708604, 0.7065453990917289, 0.7076373820553712, -0.0027774940359030264]

        startPos = [0, 0, 0.35]
        startOrn = self._pybullet_client.getQuaternionFromEuler([0, 0, math.pi * 0.5])

        flags = self._pybullet_client.URDF_MAINTAIN_LINK_ORDER + self._pybullet_client.URDF_USE_SELF_COLLISION + \
            self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

        model_path = "x_leopard_description/urdf/model_standard.urdf"
            
        self._sim_model = self._pybullet_client.loadURDF(model_path,
                                                         startPos,
                                                         startOrn,
                                                         globalScaling=1.0,
                                                         flags=flags,
                                                         useFixedBase=False)

        #self._pybullet_client.resetBasePositionAndOrientation(self._sim_model, startPos, startOrn)

        ########################################################################################################################
        useConstraints = False
        if not useConstraints:
            for j in range(self._pybullet_client.getNumJoints(self._sim_model)):
                self._pybullet_client.setJointMotorControl2(self._sim_model, j, self._pybullet_client.POSITION_CONTROL,
                                                            force=0)

        self.jointIds = []
        for j in range(self._pybullet_client.getNumJoints(self._sim_model)):
            self._pybullet_client.changeDynamics(
                self._sim_model, j, linearDamping=0, angularDamping=0)
            info = self._pybullet_client.getJointInfo(self._sim_model, j)
            print("info:", info)
            jointType = info[2]
            if (jointType == self._pybullet_client.JOINT_PRISMATIC or jointType == self._pybullet_client.JOINT_REVOLUTE):
                self.jointIds.append(j)

        print("==================================self.jointIds==================================:", self.jointIds)

        self._end_effectors = [2, 3, 5, 6, 8, 9, 11, 12]


        self._kin_model = self._pybullet_client.loadURDF(model_path,
            startPos,
            startOrn,
            globalScaling=1.0,
            useFixedBase=True,
            flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER)
        self._cycleOffset = None
        # self._pybullet_client.resetBasePositionAndOrientation(self._kin_model, startPos, startOrn)

        # ####################################################
        self.jointOffsets = []
        for _ in range(4):
            self.jointOffsets.append(0)
            self.jointOffsets.append(0)
            self.jointOffsets.append(0)
        #
        self.jointDirections = [1, -1, -1,
                                1, -1, -1,
                                1, -1, -1,
                                1, -1, -1]
        # self.jointDirections = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # #
        startQ = [
            0.08389, 0.8482, -1.547832, -0.068933, 0.625726, -
            1.272086, 0.074398, 0.61135, -1.255892,
            -0.068262, 0.836745, -1.534517
        ]
        for j in range(len(self.jointIds)):
            self._pybullet_client.resetJointState(self._sim_model, self.jointIds[j],
                                                  self.jointDirections[j] * startQ[j] + self.jointOffsets[j])
        for j in range(len(self.jointIds)):
            self._pybullet_client.resetJointState(self._kin_model, self.jointIds[j],
                                                  self.jointDirections[j] * startQ[j] + self.jointOffsets[j])

        self._pybullet_client.changeDynamics(
            self._sim_model, -1, lateralFriction=0.9)
        for j in range(self._pybullet_client.getNumJoints(self._sim_model)):
            self._pybullet_client.changeDynamics(
                self._sim_model, j, lateralFriction=0.9)

        self._pybullet_client.changeDynamics(
            self._sim_model, -1, linearDamping=0, angularDamping=0)
        ########################################################################################################################

        self._pybullet_client.changeDynamics(
            self._kin_model, -1, linearDamping=0, angularDamping=0)

        # todo: add feature to disable simulation for a particular object. Until then, disable all collisions
        self._pybullet_client.setCollisionFilterGroupMask(self._kin_model,
                                                          -1,
                                                          collisionFilterGroup=0,
                                                          collisionFilterMask=0)
        self._pybullet_client.changeDynamics(
            self._kin_model,
            -1,
            activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
            self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
            self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
        alpha = 0.4
        self._pybullet_client.changeVisualShape(
            self._kin_model, -1, rgbaColor=[1, 1, 1, alpha])
        for j in range(self._pybullet_client.getNumJoints(self._kin_model)):
            self._pybullet_client.setCollisionFilterGroupMask(self._kin_model,
                                                              j,
                                                              collisionFilterGroup=0,
                                                              collisionFilterMask=0)
            self._pybullet_client.changeDynamics(
                self._kin_model,
                j,
                activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
                self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
            self._pybullet_client.changeVisualShape(
                self._kin_model, j, rgbaColor=[1, 1, 1, alpha])

        self._poseInterpolator = leopardPoseInterpolator.LeopardPoseInterpolator()
        self.pose_package = PosePackage(self._poseInterpolator, self.noise_scale)

        ####################################################################################################################
        # self._stablePD = pd_controller_stable.PDControllerStableMultiDof(self._pybullet_client)  # 原始的代码
        self._stablePD = pd_controller_stable.PDControllerStable(
            self._pybullet_client)  # 按照testLaikago.py的代码
        ####################################################################################################################

        self._timeStep = timeStep

        self._kpOrg = [600] * 18

        self._kdOrg = [12] * 18

        self._jointIndicesAll = [
            FR_hip_motor, FR_upper_leg, FR_lower_leg,
            FL_hip_motor, FL_upper_leg, FL_lower_leg,
            RR_hip_motor, RR_upper_leg, RR_lower_leg,
            RL_hip_motor, RL_upper_leg, RL_lower_leg
        ]
        for j in self._jointIndicesAll:
            # self._pybullet_client.setJointMotorControlMultiDof(self._sim_model, j, self._pybullet_client.POSITION_CONTROL, force=[1,1,1])
            self._pybullet_client.setJointMotorControl2(self._sim_model,
                                                        j,
                                                        self._pybullet_client.POSITION_CONTROL,
                                                        targetPosition=0,
                                                        positionGain=0,
                                                        targetVelocity=0,
                                                        force=jointFrictionForce)
            # 球关节用的控制方法
            self._pybullet_client.setJointMotorControlMultiDof(
                self._sim_model,
                j,
                self._pybullet_client.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[jointFrictionForce, jointFrictionForce, jointFrictionForce])
            self._pybullet_client.setJointMotorControl2(self._kin_model,
                                                        j,
                                                        self._pybullet_client.POSITION_CONTROL,
                                                        targetPosition=0,
                                                        positionGain=0,
                                                        targetVelocity=0,
                                                        force=0)
            # 球关节用的控制方法
            self._pybullet_client.setJointMotorControlMultiDof(
                self._kin_model,
                j,
                self._pybullet_client.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[jointFrictionForce, jointFrictionForce, 0])

        # self._jointDofCounts = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]
        self._jointDofCounts = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # only those body parts/links are allowed to touch the ground, otherwise the episode terminates
        fall_contact_bodies = []
        if self._arg_parser is not None:
            fall_contact_bodies = self._arg_parser.parse_ints(
                "fall_contact_bodies")
        self._fall_contact_body_parts = fall_contact_bodies

        print("fall_contact_bodies:", fall_contact_bodies)

        # [x,y,z] base position and [x,y,z,w] base orientation!
        self._totalDofs = 7
        for dof in self._jointDofCounts:
            self._totalDofs += dof
        # print("in file laikago_stable_pd.py line 263:")
        self.setSimTime(0)

        self.resetPose()

        print("==============================laikago_stable_pd __init__ is done======================")

    def resetPose(self):
        # print("resetPose with self._frame=", self._frame, " and self._frameFraction=",self._frameFraction)
        pose = self.computePose(self._frameFraction)

        # add noise when initialize sim model;
        # but do not add noise on the kin_model;
        if self.noise_scale > 0.0:
            self.pose_package.update()
            self.initializePose(self.pose_package,
                                self._sim_model, initBase=True)  # 初始化，让狗的脚弯曲起来
        else:
            self.initializePose(self._poseInterpolator,
                                self._sim_model, initBase=True)
        self.initializePose(self._poseInterpolator,
                            self._kin_model, initBase=True)  # 初始化，让狗的脚弯曲起来


    def return_sim_pose(self):
        basePos, baseOrn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        baseLinVel, baseAngVel = self._pybullet_client.getBaseVelocity(self._sim_model)
        joint_states = self._pybullet_client.getJointStates(self._sim_model, joint_indices)
        
        jointPositions = [state[0] for state in joint_states]
        jointVelocities = [state[1] for state in joint_states]

        return basePos, baseOrn, baseLinVel, baseAngVel, jointPositions, jointVelocities


    def return_kin_pose(self):
        pose = self._poseInterpolator
        basePos, baseOrn = pose._basePos, pose._baseOrn
        baseLinVel, baseAngVel = pose._baseLinVel, pose._baseAngVel
        jointPositions = [pose._FR_hip_motorRot, pose._FR_upper_legRot, pose._FR_lower_legRot,
                            pose._FL_hip_motorRot, pose._FL_upper_legRot, 
                                pose._FL_lower_legRot,
                            pose._RR_hip_motorRot, pose._RR_upper_legRot, 
                                pose._RR_lower_legRot,
                            pose._RL_hip_motorRot, pose._RL_upper_legRot, pose._RL_lower_legRot]
        jointVelocities = [pose._FR_hip_motorVel, pose._FR_upper_legVel, pose._FR_lower_legVel,
                            pose._FL_hip_motorVel, pose._FL_upper_legVel,
                                pose._FL_lower_legVel,
                            pose._RR_hip_motorVel, pose._RR_upper_legVel,
                                pose._RR_lower_legVel,
                            pose._RL_hip_motorVel, pose._RL_upper_legVel, pose._RL_lower_legVel]

        return [np.squeeze(np.array(value)) for value in [basePos, baseOrn, baseLinVel, baseAngVel, jointPositions, jointVelocities]]


    # pass
    def initializePose(self, pose, phys_model, initBase, initializeVelocity=True):

        if initBase:
            self._pybullet_client.resetBasePositionAndOrientation(
                phys_model, pose._basePos, pose._baseOrn)
            self._pybullet_client.resetBaseVelocity(
                phys_model, pose._baseLinVel, pose._baseAngVel)
        indices = [
            FR_hip_motor, FR_upper_leg, FR_lower_leg,
            FL_hip_motor, FL_upper_leg, FL_lower_leg,
            RR_hip_motor, RR_upper_leg, RR_lower_leg,
            RL_hip_motor, RL_upper_leg, RL_lower_leg
        ]
        jointPositions = [[pose._FR_hip_motorRot], [pose._FR_upper_legRot], [pose._FR_lower_legRot],
                            [pose._FL_hip_motorRot], [pose._FL_upper_legRot], [
                                pose._FL_lower_legRot],
                            [pose._RR_hip_motorRot], [pose._RR_upper_legRot], [
                                pose._RR_lower_legRot],
                            [pose._RL_hip_motorRot], [pose._RL_upper_legRot], [pose._RL_lower_legRot]]

        jointVelocities = [[pose._FR_hip_motorVel], [pose._FR_upper_legVel], [pose._FR_lower_legVel],
                            [pose._FL_hip_motorVel], [pose._FL_upper_legVel], [
                                pose._FL_lower_legVel],
                            [pose._RR_hip_motorVel], [pose._RR_upper_legVel], [
                                pose._RR_lower_legVel],
                            [pose._RL_hip_motorVel], [pose._RL_upper_legVel], [pose._RL_lower_legVel]]

        self._pybullet_client.resetJointStatesMultiDof(
            phys_model, indices, jointPositions, jointVelocities)
            
            
    def getCycleTime(self):
        keyFrameDuration = self._mocap_data.KeyFrameDuraction()
        cycleTime = keyFrameDuration * (self._mocap_data.NumFrames() - 1)
        return cycleTime

    def setSimTime(self, t):
        self._simTime = t
        keyFrameDuration = self._mocap_data.KeyFrameDuraction()
        cycleTime = self._mocap_data.KeyFrameDuraction() * (self._mocap_data.NumFrames() - 1)
        self._cycleCount = math.floor(t / cycleTime)
        frameTime = t - self._cycleCount * cycleTime
        if (frameTime < 0):
            frameTime += cycleTime

        self._frame = int(frameTime / keyFrameDuration)

        self._frameNext = self._frame + 1
        if (self._frameNext >= self._mocap_data.NumFrames()):
            self._frameNext = self._frame

        self._frameFraction = (frameTime - self._frame *
                               keyFrameDuration) / (keyFrameDuration)

    def computeCycleOffset(self):
        """ motionData 中的数据跑完一个循环， base基点位置(position) 的偏移"""
        firstFrame = 0
        lastFrame = self._mocap_data.NumFrames() - 1
        frameData = self._mocap_data._motion_data['Frames'][0]
        frameDataNext = self._mocap_data._motion_data['Frames'][lastFrame]

        basePosStart = [frameData[1], frameData[2], frameData[3]]
        basePosEnd = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
        self._cycleOffset = [
            basePosEnd[0] - basePosStart[0], basePosEnd[1] - basePosStart[1],
            basePosEnd[2] - basePosStart[2]
        ]
        return self._cycleOffset

    def computePose(self, frameFraction):
        frameData = self._mocap_data._motion_data['Frames'][self._frame]
        frameDataNext = self._mocap_data._motion_data['Frames'][self._frameNext]
        ####################################################################################################################
        self._poseInterpolator.Slerp(
            frameFraction, frameData, frameDataNext, self._pybullet_client)

        if self.periodic:
            if self._cycleOffset is None:       # avoid dumplicating
                self.computeCycleOffset()
            
            oldPos = self._poseInterpolator._basePos
            self._poseInterpolator._basePos = [
                oldPos[0] + self._cycleCount * self._cycleOffset[0],
                oldPos[1] + self._cycleCount * self._cycleOffset[1],
                oldPos[2] + self._cycleCount * self._cycleOffset[2]
            ]
        pose = self._poseInterpolator.GetPose()

        return pose

    def convertActionToPose(self, action):
        pose = self._poseInterpolator.ConvertFromAction(
            self._pybullet_client, action)
        return pose

    def computePDForces(self, desiredPositions, desiredVelocities, maxForces):
        if desiredVelocities == None:
            ##################################################################################################################
            # desiredVelocities = [0] * self._totalDofs
            # desiredVelocities = [0] * 19
            desiredVelocities = [0] * 18
            ##################################################################################################################

        taus = self._stablePD.computePD(bodyUniqueId=self._sim_model,  # 按照普通PD控制的 testLaikago.py的配置
                                        jointIndices=self.jointIds,
                                        desiredPositions=desiredPositions,
                                        desiredVelocities=desiredVelocities,
                                        kps=self._kpOrg,
                                        kds=self._kdOrg,
                                        maxForces=maxForces,
                                        timeStep=self._timeStep)

        # print(taus)
        # print("================ taus in file laikago_stable_pd.py :", np.array(taus).shape, taus)
        return taus

    def applyPDForces(self, taus):
        """ 按照 普通 PD 控制的 testLaikago.py 的配置"""
        "joint 0 is head, can not be applied force"
        dofIndex = 6
        scaling = 1
        for index in range(len(self.jointIds)):
            jointIndex = self.jointIds[index]
            force = [scaling * taus[dofIndex]]
            self._pybullet_client.setJointMotorControlMultiDof(self._sim_model,
                                                               jointIndex,
                                                               controlMode=self._pybullet_client.TORQUE_CONTROL,
                                                               force=force)
            dofIndex += 1

    def getFrameNumber(self,):
        return self._simTime / self._mocap_data.KeyFrameDuraction()


    def getPhase(self, mod=True):
        keyFrameDuration = self._mocap_data.KeyFrameDuraction()
        cycleTime = keyFrameDuration * (self._mocap_data.NumFrames() - 1)
        phase = self._simTime / cycleTime
        if mod:
            phase = math.fmod(phase, 1.0)
            if (phase < 0):
                phase += 1
        return phase

    def buildHeadingTrans(self, rootOrn):
        # align root transform 'forward' with world-space x axis
        eul = self._pybullet_client.getEulerFromQuaternion(rootOrn)
        refDir = [1, 0, 0]
        rotVec = self._pybullet_client.rotateVector(rootOrn, refDir)
        heading = math.atan2(-rotVec[2], rotVec[0])
        heading2 = eul[1]
        # print("heading=",heading)
        headingOrn = self._pybullet_client.getQuaternionFromAxisAngle(
            [0, 1, 0], -heading)
        return headingOrn

    def buildOriginTrans(self):
        rootPos, rootOrn = self._pybullet_client.getBasePositionAndOrientation(
            self._sim_model)
        invRootPos = [-rootPos[0], 0, -rootPos[2]]
        headingOrn = self.buildHeadingTrans(rootOrn)
        invOrigTransPos, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0, 0, 0],
                                                                                    headingOrn,
                                                                                    invRootPos,
                                                                                    [0, 0, 0, 1])
        return invOrigTransPos, invOrigTransOrn

    def getBasePosAndOrt(self):
        sim_pos, sim_rot = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        kin_pos, kin_rot = self._pybullet_client.getBasePositionAndOrientation(self._kin_model)
        return sim_pos, sim_rot, kin_pos, kin_rot

    def get_traj_data(self):
        traj_data = []
        for i in range(self._mocap_data.NumFrames()):
            traj_data.append(self.get_points(i))
        
        return np.array(traj_data)

    def get_points(self, cur_frame):
        local_traj = []

        this_pose = self._mocap_data._motion_data['Frames'][cur_frame][1:4]
        this_ort = self._mocap_data._motion_data['Frames'][cur_frame][4:8]

        for shift in [i * self.ds_step for i in range(1, self.view_rad+1)]:
            front_frame = self._mocap_data._motion_data['Frames'][max(0, cur_frame - shift)]
            rear_frame = self._mocap_data._motion_data['Frames'][min(self._mocap_data.NumFrames() - 1, cur_frame + shift)]

            front_next_frame = self._mocap_data._motion_data['Frames'][max(0, cur_frame - shift + 1)]
            rear_next_frame = self._mocap_data._motion_data['Frames'][min(self._mocap_data.NumFrames() - 1, cur_frame + shift + 1)]

            if self.tm_args.points:
                front_pose = [front_frame[i + 1] - this_pose[i] for i in range(3)]
                rear_pose = [rear_frame[i + 1] - this_pose[i] for i in range(3)]
            else:
                rear_pose = front_pose = []

            if self.tm_args.use_ort:                    
                front_ort = front_frame[4:8]
                rear_ort = rear_frame[4:8]
            else:
                rear_ort = front_ort = []

            local_traj = rear_pose + rear_ort + local_traj + front_pose + front_ort

            if self.tm_args.velocity:
                front_vxy = [(front_next_frame[i] - front_frame[i]) / 0.01667 for i in range(1, 3)]
                rear_vxy = [(rear_next_frame[i] - rear_frame[i]) / 0.01667 for i in range(1, 3)]

                local_traj = front_vxy + local_traj + rear_vxy

        return local_traj

        
    def getState(self):

        self.stateVector = []
        
        if self.phase_instr == 'normal':
            phase = self.getPhase()
            self.stateVector.append(phase)
        elif self.phase_instr == 'replace':
            # if self.tm_args.points:
            self.stateVector += self.get_points(self._frame)
        else:
            raise NotImplementedError

        rootTransPos, rootTransOrn = self.buildOriginTrans()
        basePos, baseOrn = self._pybullet_client.getBasePositionAndOrientation(
            self._sim_model)

        # rootPosRel = [0, ?, 0]
        rootPosRel, dummy = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                     basePos, [0, 0, 0, 1])
        
        if self.phase_instr == 'normal':
            self.stateVector.append(rootPosRel[1])
        else:
            self.rootPosRel_backup = rootPosRel[1]

        linkStatesSim = self._pybullet_client.getLinkStates(self._sim_model, self.jointIds[:12],
                                                            computeForwardKinematics=True, computeLinkVelocity=True)

        # 12 * (3 + 4) = 84
        for pbJoint in range(12):
            j = self.jointIds[pbJoint]
            ls = linkStatesSim[pbJoint]
            linkPos = ls[0]
            linkOrn = ls[1]

            '''compute original frame'''
            linkPosLocal, linkOrnLocal = self._pybullet_client.multiplyTransforms(
                rootTransPos, rootTransOrn, linkPos, linkOrn)
            if (linkOrnLocal[3] < 0):
                linkOrnLocal = [-linkOrnLocal[0], -linkOrnLocal[1], -linkOrnLocal[2], -linkOrnLocal[3]]
            linkPosLocal = [
                linkPosLocal[0] -
                rootPosRel[0], linkPosLocal[1] - rootPosRel[1],
                linkPosLocal[2] - rootPosRel[2]
            ]
            for l in linkPosLocal:
                self.stateVector.append(l)
            # re-order the quaternion, DeepMimic uses w,x,y,z
            self.stateVector += [
                linkOrnLocal[3], linkOrnLocal[0],
                linkOrnLocal[1], linkOrnLocal[2]
            ]



        # 12 * 6 = 72
        for pbJoint in range(12):
            j = self.jointIds[pbJoint]
            # ls = self._pybullet_client.getLinkState(self._sim_model, j, computeLinkVelocity=True)
            ls = linkStatesSim[pbJoint]

            linkLinVel = ls[6]
            linkAngVel = ls[7]

            linkLinVelLocal, _ = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                               linkLinVel, [0, 0, 0, 1])
            # linkLinVelLocal=[linkLinVelLocal[0]-rootPosRel[0],linkLinVelLocal[1]-rootPosRel[1],linkLinVelLocal[2]-rootPosRel[2]]
            linkAngVelLocal, _ = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                               linkAngVel, [0, 0, 0, 1])

            '''original '''
            self.stateVector += linkLinVelLocal
            self.stateVector += linkAngVelLocal

        state = self.stateVector

        return state

    def terminates(self):
        # check if any non-allowed body part hits the ground
        terminates = False
        pts = self._pybullet_client.getContactPoints()
        
        for p in pts:
            part = -1
            # ignore self-collision
            if (p[1] == p[2]):
                continue
            if (p[1] == self._sim_model):
                part = p[3]
            if (p[2] == self._sim_model):
                part = p[4]
            if (part >= 0 and part in self._fall_contact_body_parts):
                # print("terminating part:", part)
                terminates = True
        ############################################################################################################
        # 增加后仰超过 90度的时候的时候，提前终止
        basePos, baseOrn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        Euler = self._pybullet_client.getEulerFromQuaternion(baseOrn)
        if (Euler[0] > (1.56 + 3.14)) or (Euler[0] < (1.56-3.14)):   # 正常行走的时候是 (3.14 / 2)
            # print("Terminates because of Euler 0", Euler)
            # time.sleep(0.5)
            terminates = True
        if (Euler[1] > (1.56 + 2.76)) or (Euler[1] < (1.56-2.76)):   
            # print("Terminates because of Euler 1", Euler)
            # time.sleep(0.5)
            terminates = True

        #############################################################################################################
        # Add termination if the position shift is too large
        kinPos, _ = self._pybullet_client.getBasePositionAndOrientation(self._kin_model)
        simPos, _ = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
        if np.sum(np.square(np.array(kinPos)-np.array(simPos))) > self.tm_args.pos_diff:
            # print('Terminates because of pos diff')
            terminates = True

        if not self.periodic:       # if not periodic, then we must terminate within one period            
            phase = self.getPhase(mod=False)
            if phase >= 1.0:
                terminates = True
                # print('Terminates because of periodic')
        
        Toe_Pos1 = self._pybullet_client.getLinkState(self._sim_model, 13)[0]
        Toe_Pos2 = self._pybullet_client.getLinkState(self._sim_model, 14)[0]        
        if Toe_Pos1[-1] > self.tm_args.toe and Toe_Pos2[-1] > self.tm_args.toe:
            terminates = True

        return terminates

    def quatMul(self, q1, q2):
        return [
            q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
            q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2],
            q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0],
            q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
        ]

    def calcRootAngVelErr(self, vel0, vel1):
        diff = [vel0[0] - vel1[0], vel0[1] - vel1[1], vel0[2] - vel1[2]]
        return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

    def calcRootRotDiff(self, orn0, orn1):
        orn0Conj = [-orn0[0], -orn0[1], -orn0[2], orn0[3]]
        q_diff = self.quatMul(orn1, orn0Conj)
        axis, angle = self._pybullet_client.getAxisAngleFromQuaternion(q_diff)
        return angle * angle

    def get_kin_model_position(self):
        return self._pybullet_client.getBasePositionAndOrientation(self._kin_model)

    def getReward(self):
        """ 参考论文 Learning Agile Robotic Locomotion Skills by Imitating Animals """
        # todo: compensate for ground height in some parts, once we move to non-flat terrain
        pose_w = 0.5
        vel_w = 0.1
        end_eff_w = 0.2
        root_w = 0.2
        com_w = 0  # 0.1
        #energy_w = 0.1

        # total_w = pose_w + vel_w + end_eff_w + root_w + com_w
        total_w = pose_w + vel_w + end_eff_w + root_w + com_w  # + energy_w
        pose_w /= total_w
        vel_w /= total_w
        end_eff_w /= total_w
        root_w /= total_w
        com_w /= total_w
        #energy_w /= total_w

        pose_scale = 5  # 5     # 2
        vel_scale = 0.5  # 0.5    # 0.1
        end_eff_scale = 40
        root_scale = 5
        com_scale = 10

        # energy_scale = 15 # -------------------------------------------------

        err_scale = 1

        reward = 0

        pose_err = 0
        vel_err = 0
        end_eff_err = 0
        root_err = 0
        com_err = 0
        heading_err = 0

        #energy_err = 0

        root_id = 0

        mJointWeights = [
            0.02333, 0.14333-0.01, 0.08333,
            0.02333, 0.14333-0.01, 0.08333,
            0.02333, 0.14333-0.01, 0.08333,
            0.02333, 0.14333-0.01, 0.08333
        ]

        num_end_effs = 0
        num_joints = 12

        # root_rot_w = mJointWeights[root_id]
        root_rot_w = 0.05

        rootPosSim, rootOrnSim = self._pybullet_client.getBasePositionAndOrientation(
            self._sim_model)
        rootPosKin, rootOrnKin = self._pybullet_client.getBasePositionAndOrientation(
            self._kin_model)
        linVelSim, angVelSim = self._pybullet_client.getBaseVelocity(
            self._sim_model)
        # don't read the velocities from the kinematic model (they are zero), use the pose interpolator velocity
        # see also issue https://github.com/bulletphysics/bullet3/issues/2401
        linVelKin = self._poseInterpolator._baseLinVel
        angVelKin = self._poseInterpolator._baseAngVel

        root_rot_err = self.calcRootRotDiff(rootOrnSim, rootOrnKin)
        pose_err += root_rot_w * root_rot_err

        root_vel_diff = [
            linVelSim[0] - linVelKin[0], linVelSim[1] -
            linVelKin[1], linVelSim[2] - linVelKin[2]
        ]
        root_vel_err = root_vel_diff[0] * root_vel_diff[0] + root_vel_diff[1] * root_vel_diff[
            1] + root_vel_diff[2] * root_vel_diff[2]

        root_ang_vel_err = self.calcRootAngVelErr(angVelSim, angVelKin)
        vel_err += root_rot_w * root_ang_vel_err

        useArray = True

        if useArray:
            jointIndices = range(num_joints)
            simJointStates = self._pybullet_client.getJointStatesMultiDof(
                self._sim_model, jointIndices)
            kinJointStates = self._pybullet_client.getJointStatesMultiDof(
                self._kin_model, jointIndices)
            # print("kinJointStates:", kinJointStates)
        if useArray:
            linkStatesSim = self._pybullet_client.getLinkStates(
                self._sim_model, jointIndices)
            linkStatesKin = self._pybullet_client.getLinkStates(
                self._kin_model, jointIndices)
        for j in range(num_joints):
            curr_pose_err = 0
            curr_vel_err = 0
            w = mJointWeights[j]
            if useArray:
                simJointInfo = simJointStates[j]
            else:
                simJointInfo = self._pybullet_client.getJointStateMultiDof(
                    self._sim_model, j)

            if useArray:
                kinJointInfo = kinJointStates[j]
            else:
                kinJointInfo = self._pybullet_client.getJointStateMultiDof(
                    self._kin_model, j)
            if (len(simJointInfo[0]) == 1):
                angle = simJointInfo[0][0] - kinJointInfo[0][0]
                curr_pose_err = angle * angle
                velDiff = simJointInfo[1][0] - kinJointInfo[1][0]
                curr_vel_err = velDiff * velDiff

            if (len(simJointInfo[0]) == 4):
                diffQuat = self._pybullet_client.getDifferenceQuaternion(
                    simJointInfo[0], kinJointInfo[0])
                axis, angle = self._pybullet_client.getAxisAngleFromQuaternion(
                    diffQuat)
                curr_pose_err = angle * angle
                diffVel = [
                    simJointInfo[1][0] -
                    kinJointInfo[1][0], simJointInfo[1][1] -
                    kinJointInfo[1][1],
                    simJointInfo[1][2] - kinJointInfo[1][2]
                ]
                curr_vel_err = diffVel[0] * diffVel[0] + \
                    diffVel[1] * diffVel[1] + diffVel[2] * diffVel[2]

            pose_err += w * curr_pose_err
            vel_err += w * curr_vel_err
            # energy_err += w * cur_energy_err

            is_end_eff = j in self._end_effectors

            if is_end_eff:

                if useArray:
                    linkStateSim = linkStatesSim[j]
                    linkStateKin = linkStatesKin[j]
                else:
                    linkStateSim = self._pybullet_client.getLinkState(
                        self._sim_model, j)
                    linkStateKin = self._pybullet_client.getLinkState(
                        self._kin_model, j)
                linkPosSim = linkStateSim[0]
                linkPosKin = linkStateKin[0]
                linkPosDiff = [
                    linkPosSim[0] - linkPosKin[0], linkPosSim[1] -
                    linkPosKin[1],
                    linkPosSim[2] - linkPosKin[2]
                ]
                curr_end_err = linkPosDiff[0] * linkPosDiff[0] + linkPosDiff[1] * linkPosDiff[
                    1] + linkPosDiff[2] * linkPosDiff[2]
                end_eff_err += curr_end_err
                num_end_effs += 1

        if (num_end_effs > 0):
            end_eff_err /= num_end_effs

        # double root_ground_h0 = mGround->SampleHeight(sim_char.GetRootPos())
        # double root_ground_h1 = kin_char.GetOriginPos()[1]
        # root_pos0[1] -= root_ground_h0
        # root_pos1[1] -= root_ground_h1
        root_pos_diff = [
            rootPosSim[0] - rootPosKin[0], rootPosSim[1] -
            rootPosKin[1], rootPosSim[2] - rootPosKin[2]
        ]
        root_pos_err = root_pos_diff[0] * root_pos_diff[0] + root_pos_diff[1] * root_pos_diff[
            1] + root_pos_diff[2] * root_pos_diff[2]
        #
        # root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1)
        # root_rot_err *= root_rot_err

        # root_vel_err = (root_vel1 - root_vel0).squaredNorm()
        # root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm()

        root_err = root_pos_err + 0.1 * root_rot_err + \
            0.01 * root_vel_err + 0.001 * root_ang_vel_err

        # com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm()

        # print("pose_err=",pose_err)
        # print("vel_err=",vel_err)
        pose_reward = math.exp(-err_scale * pose_scale * pose_err)
        vel_reward = math.exp(-err_scale * vel_scale * vel_err)
        end_eff_reward = math.exp(-err_scale * end_eff_scale * end_eff_err)
        root_reward = math.exp(-err_scale * root_scale * root_err)
        com_reward = math.exp(-err_scale * com_scale * com_err)
        # energy_reward = math.exp(-err_scale * energy_scale * energy_err)

        # reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward + root_w * root_reward + com_w * com_reward
        reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward + \
            root_w * root_reward + com_w * com_reward  # + energy_w * energy_reward

        # pose_reward,vel_reward,end_eff_reward, root_reward, com_reward);
        # print("reward=",reward)
        # print("pose_reward=",pose_w * pose_reward)
        # print("vel_reward=",vel_w * vel_reward)
        # print("end_eff_reward=",end_eff_w * end_eff_reward)
        # print("root_reward=",root_w * root_reward)
        # print("com_reward=",com_w * com_reward)
        # print("energy_reward=", energy_w * energy_reward)

        return reward
