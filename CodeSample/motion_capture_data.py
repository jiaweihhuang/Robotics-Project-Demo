import json
import math
from copy import deepcopy

class MotionCaptureData(object):

  def __init__(self):
    self.Reset()

  def Reset(self):
    self._motion_data = []

  def Load(self, path, startFrame=None, endFrame=None, revisedRM=None):
    with open(path, 'r') as f:
      self._all_motion_data = json.load(f)
      if revisedRM is not None:
        self._all_motion_data['Frames'] = revisedRM['Frames']
      self._motion_data = deepcopy(self._all_motion_data)
      if startFrame is not None and endFrame is not None:
        assert endFrame > startFrame
        self._motion_data['Frames'] = self._all_motion_data['Frames'][startFrame:endFrame]

  def reset_motion_data(self, startFrame, endFrame):
    self._motion_data['Frames'] = self._all_motion_data['Frames'][startFrame:endFrame]

  def appendDuration2Frames(self, duration):
      # print(self._motion_data)
      assert self._motion_data is not None, "Haven't load data"
      if len(self._motion_data['Frames'][0]) == 20:         # if we already have one, then do nothing
          return
      else:
          assert len(self._motion_data['Frames'][0]) == 19, 'Length of Frame Data is incorrect'
          
      for f in self._motion_data['Frames']:
          f.insert(0, duration)
          
  def NumFrames(self):
    return len(self._motion_data['Frames'])

  def KeyFrameDuraction(self):
    return self._motion_data['Frames'][0][0]

  def getCycleTime(self):
    keyFrameDuration = self.KeyFrameDuraction()
    cycleTime = keyFrameDuration * (self.NumFrames() - 1)
    return cycleTime

  def calcCycleCount(self, simTime, cycleTime):
    phases = simTime / cycleTime
    count = math.floor(phases)
    loop = True
    #count = (loop) ? count : cMathUtil::Clamp(count, 0, 1);
    return count

  def computeCycleOffset(self):
    firstFrame = 0
    lastFrame = self.NumFrames() - 1
    frameData = self._motion_data['Frames'][0]
    frameDataNext = self._motion_data['Frames'][lastFrame]

    basePosStart = [frameData[1], frameData[2], frameData[3]]
    basePosEnd = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
    self._cycleOffset = [
        basePosEnd[0] - basePosStart[0], basePosEnd[1] - basePosStart[1],
        basePosEnd[2] - basePosStart[2]
    ]
    return self._cycleOffset
