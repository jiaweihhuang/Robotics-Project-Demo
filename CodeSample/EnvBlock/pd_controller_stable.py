import numpy as np


class PDControllerStable(object):

  def __init__(self, pb):
    self._pb = pb

  def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds,
                maxForces, timeStep):
    numBaseDofs = 0
    numPosBaseDofs = 0
    baseMass = self._pb.getDynamicsInfo(bodyUniqueId, -1)[0]
    curPos, curOrn = self._pb.getBasePositionAndOrientation(bodyUniqueId)
    q1 = []
    qdot1 = []
    zeroAccelerations = []
    qError = []
    if (baseMass > 0):
      numBaseDofs = 6
      numPosBaseDofs = 7
      q1 = [curPos[0], curPos[1], curPos[2], curOrn[0], curOrn[1], curOrn[2], curOrn[3]]
      qdot1 = [0] * numBaseDofs
      zeroAccelerations = [0] * numBaseDofs
      angDiff = [0, 0, 0]
      qError = [
          desiredPositions[0] - curPos[0], desiredPositions[1] - curPos[1],
          desiredPositions[2] - curPos[2], angDiff[0], angDiff[1], angDiff[2]
      ]
    numJoints = 12
    jointStates = self._pb.getJointStates(bodyUniqueId, jointIndices)

    for i in range(numJoints):
      q1.append(jointStates[i][0])
      qdot1.append(jointStates[i][1])
      zeroAccelerations.append(0)
    q = np.array(q1)
    qdot = np.array(qdot1)
    qdes = np.array(desiredPositions)
    qdotdes = np.array(desiredVelocities)
    #qError = qdes - q
    for j in range(numJoints):
      qError.append(desiredPositions[j + numPosBaseDofs] - q1[j + numPosBaseDofs])
    #print("qError=",qError)
    qdotError = qdotdes - qdot
    Kp = np.diagflat(kps)
    Kd = np.diagflat(kds)
    p = Kp.dot(qError)
    d = Kd.dot(qdotError)
    forces = p + d

    M1 = self._pb.calculateMassMatrix(bodyUniqueId, q1)
    M2 = np.array(M1)
    M = (M2 + Kd * timeStep)
    c1 = self._pb.calculateInverseDynamics(bodyUniqueId, q1, qdot1, zeroAccelerations)
    c = np.array(c1)
    A = M
    b = -c + p + d
    qddot = np.linalg.solve(A, b)
    tau = p + d - Kd.dot(qddot) * timeStep
    maxF = np.array(maxForces)
    tau = np.clip(tau, -maxF, maxF)
    #print("c=",c)
    return tau
