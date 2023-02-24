from pybullet_utils import bullet_client
import math


class LeopardPoseInterpolator(object):

    def __init__(self):
        pass

    def ComputeLinVel(self, posStart, posEnd, deltaTime):
        vel = [(posEnd[0] - posStart[0]) / deltaTime, (posEnd[1] - posStart[1]) / deltaTime,
               (posEnd[2] - posStart[2]) / deltaTime]
        return vel

    def ComputeAngVel(self, ornStart, ornEnd, deltaTime, bullet_client):
        dorn = bullet_client.getDifferenceQuaternion(ornStart, ornEnd)
        axis, angle = bullet_client.getAxisAngleFromQuaternion(dorn)
        angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
                  (axis[2] * angle) / deltaTime]
        return angVel

    def ComputeAngVelRel(self, ornStart, ornEnd, deltaTime, bullet_client):
        ornStartConjugate = [-ornStart[0], -
                             ornStart[1], -ornStart[2], ornStart[3]]
        pos_diff, q_diff = bullet_client.multiplyTransforms([0, 0, 0], ornStartConjugate, [0, 0, 0],
                                                            ornEnd)
        axis, angle = bullet_client.getAxisAngleFromQuaternion(q_diff)
        angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
                  (axis[2] * angle) / deltaTime]
        return angVel

    def Slerp(self, frameFraction, frameData, frameDataNext, bullet_client):
        assert len(frameData) == 20
        keyFrameDuration = frameData[0]
        basePos1Start = [frameData[1], frameData[2], frameData[3]]
        basePos1End = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
        self._basePos = [
            basePos1Start[0] + frameFraction *
            (basePos1End[0] - basePos1Start[0]),
            basePos1Start[1] + frameFraction *
            (basePos1End[1] - basePos1Start[1]),
            basePos1Start[2] + frameFraction * \
            (basePos1End[2] - basePos1Start[2])
        ]

        self._baseLinVel = self.ComputeLinVel(
            basePos1Start, basePos1End, keyFrameDuration)

        baseOrn1Start = [frameData[4], frameData[5], frameData[6], frameData[7]]                    # Laikago
        baseOrn1Next = [frameDataNext[4], frameDataNext[5], frameDataNext[6], frameDataNext[7]]

        self._baseOrn = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction) # Laikago

        self._baseAngVel = self.ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration,
                                              bullet_client)

        jointPositions = [
            self._basePos[0], self._basePos[1], self._basePos[2], self._baseOrn[0], self._baseOrn[1],
            self._baseOrn[2], self._baseOrn[3]
        ]
        jointVelocities = [
            self._baseLinVel[0], self._baseLinVel[1], self._baseLinVel[2], self._baseAngVel[0],
            self._baseAngVel[1], self._baseAngVel[2]
        ]

        for j in range(12):
            index = j + 8
            jointPosStart = frameData[index]
            jointPosEnd = frameDataNext[index]
            jointPos = jointPosStart + frameFraction * \
                (jointPosEnd - jointPosStart)
            jointVel = (jointPosEnd - jointPosStart) / keyFrameDuration
            jointPositions.append(jointPos)
            jointVelocities.append(jointVel)

        ####################################################################################################################
        correction = True  # # 加了修正以后, sim_model 的重心升高，动作更自然
        # jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
        # jointOffsets = []
        # for i in range(4):
        #     jointOffsets.append(0)
        #     jointOffsets.append(-0.7)
        #     jointOffsets.append(0.7)
        jointOffsets = []
        for i in range(4):
            jointOffsets.append(0)
            jointOffsets.append(0)
            jointOffsets.append(0)

        # jointDirections = [1, -1, -1,
        #                    1, -1, -1,
        #                    1, -1, -1,
        #                    1, -1, -1]
        jointDirections = [1, 1, 1,
                           1, 1, 1,
                           1, 1, 1,
                           1, 1, 1]

        '''
        ??? What is Correction used for?
        '''
        if correction:
            desiredPositions = []
            for j in range(7):
                targetPosUnmodified = float(jointPositions[j])
                desiredPositions.append(targetPosUnmodified)

            for j in range(12):
                targetPosUnmodified = float(jointPositions[j + 7])
                # 加了修正以后, sim_model 的重心升高，动作更自然
                targetPos = jointDirections[j] * \
                    targetPosUnmodified + jointOffsets[j]
                desiredPositions.append(targetPos)

            jointPositions = desiredPositions

        ####################################################################################################################
        self._jointPositions = jointPositions
        self._jointVelocities = jointVelocities
        ###################################################
        self._FR_hip_motorRot = jointPositions[7]
        self._FR_upper_legRot = jointPositions[8]
        self._FR_lower_legRot = jointPositions[9]

        self._FL_hip_motorRot = jointPositions[10]
        self._FL_upper_legRot = jointPositions[11]
        self._FL_lower_legRot = jointPositions[12]

        self._RR_hip_motorRot = jointPositions[13]
        self._RR_upper_legRot = jointPositions[14]
        self._RR_lower_legRot = jointPositions[15]

        self._RL_hip_motorRot = jointPositions[16]
        self._RL_upper_legRot = jointPositions[17]
        self._RL_lower_legRot = jointPositions[18]

        ###################################################
        self._FR_hip_motorVel = jointVelocities[6]
        self._FR_upper_legVel = jointVelocities[7]
        self._FR_lower_legVel = jointVelocities[8]

        self._FL_hip_motorVel = jointVelocities[9]
        self._FL_upper_legVel = jointVelocities[10]
        self._FL_lower_legVel = jointVelocities[11]

        self._RR_hip_motorVel = jointVelocities[12]
        self._RR_upper_legVel = jointVelocities[13]
        self._RR_lower_legVel = jointVelocities[14]

        self._RL_hip_motorVel = jointVelocities[15]
        self._RL_upper_legVel = jointVelocities[16]
        self._RL_lower_legVel = jointVelocities[17]

        ####################################################
        # print(len(jointPositions))  # 19
        # print(len(jointVelocities)) # 18
        return jointPositions, jointVelocities

    def GetPose(self):
        # print("in func GetPose")
        pose = [
            self._basePos[0], self._basePos[1], self._basePos[2],
            self._baseOrn[0], self._baseOrn[1], self._baseOrn[2], self._baseOrn[3],

            # self._FR_hip_motorRot[0],
            # self._FR_upper_legRot[0],
            # self._FR_lower_legRot[0],
            #
            # self._FL_hip_motorRot[0],
            # self._FL_upper_legRot[0],
            # self._FL_lower_legRot[0],
            #
            # self._RR_hip_motorRot[0],
            # self._RR_upper_legRot[0],
            # self._RR_lower_legRot[0],
            #
            # self._RL_hip_motorRot[0],
            # self._RL_upper_legRot[0],
            # self._RL_lower_legRot[0]

            self._FR_hip_motorRot,
            self._FR_upper_legRot,
            self._FR_lower_legRot,

            self._FL_hip_motorRot,
            self._FL_upper_legRot,
            self._FL_lower_legRot,

            self._RR_hip_motorRot,
            self._RR_upper_legRot,
            self._RR_lower_legRot,

            self._RL_hip_motorRot,
            self._RL_upper_legRot,
            self._RL_lower_legRot
        ]
        return pose

    def ConvertFromAction(self, pybullet_client, action):
        """ 与 laikago_pose_interpolator.py 中的本函数一模一样 """
        # print("in func ConvertFromAction")
        # turn action into pose

        # self.Reset()  #?? needed?
        # index = 0
        #
        # angle = action[index]
        # index += 1
        # self._FR_hip_motorRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._FR_upper_legRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._FR_lower_legRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._FL_hip_motorRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._FL_upper_legRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._FL_lower_legRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._RR_hip_motorRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._RR_upper_legRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._RR_lower_legRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._RL_hip_motorRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._RL_upper_legRot = [angle]
        #
        # angle = action[index]
        # index += 1
        # self._RL_lower_legRot = [angle]

        # self.Reset()  #?? needed?
        index = 0

        angle = action[index]
        index += 1
        self._FR_hip_motorRot = angle

        angle = action[index]
        index += 1
        self._FR_upper_legRot = angle

        angle = action[index]
        index += 1
        self._FR_lower_legRot = angle

        angle = action[index]
        index += 1
        self._FL_hip_motorRot = angle

        angle = action[index]
        index += 1
        self._FL_upper_legRot = angle

        angle = action[index]
        index += 1
        self._FL_lower_legRot = angle

        angle = action[index]
        index += 1
        self._RR_hip_motorRot = angle

        angle = action[index]
        index += 1
        self._RR_upper_legRot = angle

        angle = action[index]
        index += 1
        self._RR_lower_legRot = angle

        angle = action[index]
        index += 1
        self._RL_hip_motorRot = angle

        angle = action[index]
        index += 1
        self._RL_upper_legRot = angle

        angle = action[index]
        index += 1
        self._RL_lower_legRot = angle

        pose = self.GetPose()
        return pose


    
    '''
    def Reset(self,
              basePos=[0, 0, 0],
              baseOrn=[0, 0, 0, 1],

              FR_hip_motorRot=[0],
              FR_upper_legRot=[0],
              FR_lower_legRot=[0],

              FL_hip_motorRot=[0],
              FL_upper_legRot=[0],
              FL_lower_legRot=[0],

              RR_hip_motorRot=[0],
              RR_upper_legRot=[0],
              RR_lower_legRot=[0],

              RL_hip_motorRot=[0],
              RL_upper_legRot=[0],
              RL_lower_legRot=[0],

              baseLinVel=[0, 0, 0],
              baseAngVel=[0, 0, 0],

              FR_hip_motorVel=[0],
              FR_upper_legVel=[0],
              FR_lower_legVel=[0],

              FL_hip_motorVel=[0],
              FL_upper_legVel=[0],
              FL_lower_legVel=[0],

              RR_hip_motorVel=[0],
              RR_upper_legVel=[0],
              RR_lower_legVel=[0],

              RL_hip_motorVel=[0],
              RL_upper_legVel=[0],
              RL_lower_legVel=[0]):
        self._basePos = basePos
        self._baseLinVel = baseLinVel

        # print("HumanoidPoseInterpolator.Reset: baseLinVel = ", baseLinVel)
        self._baseOrn = baseOrn
        self._baseAngVel = baseAngVel

        self._FR_hip_motorRot = FR_hip_motorRot
        self._FR_hip_motorVel = FR_hip_motorVel

        self._FR_upper_legRot = FR_upper_legRot
        self._FR_upper_legVel = FR_upper_legVel

        self._FR_lower_legRot = FR_lower_legRot
        self._FR_lower_legVel = FR_lower_legVel

        self._FL_hip_motorRot = FL_hip_motorRot
        self._FL_hip_motorVel = FL_hip_motorVel

        self._FL_upper_legRot = FL_upper_legRot
        self._FL_upper_legVel = FL_upper_legVel

        self._FL_lower_legRot = FL_lower_legRot
        self._FL_lower_legVel = FL_lower_legVel

        self._RR_hip_motorRot = RR_hip_motorRot
        self._RR_hip_motorVel = RR_hip_motorVel

        self._RR_upper_legRot = RR_upper_legRot
        self._RR_upper_legVel = RR_upper_legVel

        self._RR_lower_legRot = RR_lower_legRot
        self._RR_lower_legVel = RR_lower_legVel

        self._RL_hip_motorRot = RL_hip_motorRot
        self._RL_hip_motorVel = RL_hip_motorVel

        self._RL_upper_legRot = RL_upper_legRot
        self._RL_upper_legVel = RL_upper_legVel

        self._RL_lower_legRot = RL_lower_legRot
        self._RL_lower_legVel = RL_lower_legVel
    '''
    