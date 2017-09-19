# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC) # Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for self.software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with self.program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""TODO"""

import collections
import csv

import numpy as np

MEMORY_LENGTH = 10000



class Experiment(object):


  def __init__(self, locationWidth, worldWidth, initialWeight, inc, dec):
    self.locWidth = locationWidth
    self.initialWeight = initialWeight
    self.inc = inc
    self.dec = dec

    # HARD-CODED ALGORITHM STATE

    # Hard-coded list of possible location offsets identified by index
    offsets = []
    for i in xrange(-(locationWidth / 2), (locationWidth / 2) + 1):
      for j in xrange(-(locationWidth / 2), (locationWidth / 2) + 1):
        if i != 0 or j != 0:
          offsets.append((i, j))
    self.offsets = offsets

    # Hard-coded mapping from motor command ID to real-world change
    self.motorMap = dict([(i, offset) for i, offset in enumerate(self.offsets)])

    # LEARNED ALGORITHM STATE

    # Learned mapping from motor command ID to [offset ID, permanence] pair
    self.offsetMap = {}
    for i in xrange(len(offsets)):
      self.addOffset(i)

    # WORKING ALGORITHM STATE

    # Current location cell
    self.loc = [0, 0]

    # Map from feature to most recent location
    self.memory = {}
    # How many instances of each feature are in the history
    self.count = collections.defaultdict(int)
    # List of features in short term memory
    self.history = []

    #self.reverseMemory = {}
    #self.reverseHistory = []
    #self.reverseCount = collections.defaultdict(int)

    # WORLD STATE

    # Current world position
    self.pos = [0, 0]

    # Build a world
    sensoryInputs = range(worldWidth ** 2)
    np.random.shuffle(sensoryInputs)
    world = np.array(sensoryInputs, dtype=int)
    world.resize((worldWidth, worldWidth))
    self.world = world

    # EXPERIMENT STATE

    self.nActive = np.zeros((self.locWidth, self.locWidth), dtype=int)
    self.stats = {
        "n": 0,
        "noPrediction": 0,
        "unknownPrediction": 0,
        "badPrediction": 0,
        "goodPrediction": 0,
    }


  def addOffset(self, i):
    possibleOffsets = (self.locWidth ** 2) - 1
    self.offsetMap[i] = [np.random.randint(possibleOffsets), self.initialWeight]


  def getNextMotor(self):
    candidates = []
    for m in self.motorMap.keys():
      xd, yd = self.motorMap[m]
      x = self.pos[0] + xd
      y = self.pos[1] + yd
      if x >= 0 and y >= 0 and x < len(self.world[0]) and y < len(self.world):
        # The first value is the square root of the permanence. The square
        # root up-weights smaller values so they still have a chance of
        # selection.
        candidates.append((self.offsetMap[m][1] ** 2, m))

    # Select a point in the cumulative weights
    totalWeights = 0.0
    for w, _ in candidates:
      totalWeights += w
    point = np.random.random() * totalWeights

    # Find which motor command this corresponds to and return it
    cum = 0.0
    for w, m in candidates:
      cum = cum + w
      if cum > point:
        return m
    raise ValueError("Should never get here...")


  def updatePos(self, currentMotor):
    xd, yd = self.motorMap[currentMotor]
    self.pos = (self.pos[0] + xd, self.pos[1] + yd)


  def updateLoc(self, currentMotor):
    offset = self.offsetMap[currentMotor][0]
    xd, yd = self.offsets[offset]
    self.loc = [v % self.locWidth for v in (self.loc[0] + xd, self.loc[1] + yd)]


  def adjustMotorWeights(self, currentMotor, newOffset):
    """

    :param newOffset: offset id of the offset that we expect to complete the cycle
    """
    currentOffset = self.offsetMap[currentMotor][0]
    if self.offsets[currentOffset] == newOffset:
      self.offsetMap[currentMotor][1] = min(self.offsetMap[currentMotor][1] + self.inc, 1.0)
    else:
      self.offsetMap[currentMotor][1] -= self.dec
      if self.offsetMap[currentMotor][1] < self.dec:
        # Delete the current mapping completely and pick a random new mapping
        del self.offsetMap[currentMotor]
        self.addOffset(currentMotor)


  def printMotor(self):
    for m, pair in self.offsetMap.iteritems():
      worldOffset = self.motorMap[m]
      locOffset = self.offsets[pair[0]]
      print "{} maps to {}".format(",".join(str(v) for v in worldOffset),
                                   ",".join(str(v) for v in locOffset))


  def printWeightStats(self):
    weights = sorted([pair[1] for pair in self.offsetMap.values()])
    median = weights[len(weights) / 2]
    total = 0.0
    minVal = 1.0
    maxVal = 0.0
    for w in weights:
      total += w
      minVal = min(minVal, w)
      maxVal = max(maxVal, w)
    mean = total / float(len(weights))
    print minVal, median, mean, maxVal


  def runOne(self):
    self.stats["n"] += 1

    currentMotor = self.getNextMotor()

    oldLoc = self.loc

    self.updatePos(currentMotor)
    self.updateLoc(currentMotor)

    feat = self.world[self.pos[0]][self.pos[1]]

    if self.count[feat] > 0:
      # We have a cycle!

      #print "cycle!"
      if self.loc == self.memory[feat]:
        self.stats["goodPrediction"] += 1
      else:
        self.stats["badPrediction"] += 1

      # If our location doesn't match the expected
      self.loc = self.memory[feat]

      # Calculate what offset would have completed the cycle in location layer
      newOffset = (self.loc[0] - oldLoc[0], self.loc[1] - oldLoc[1])
      # Decrement the motor offset mapping if the cycle wasn't correct
      self.adjustMotorWeights(currentMotor, newOffset)
    else:
      self.stats["unknownPrediction"] += 1

    self.nActive[self.loc[0]][self.loc[1]] += 1

    # Add new feature to history, etc
    self.history.append(feat)
    self.count[feat] += 1
    self.memory[feat] = self.loc

    #self.reverseHistory.append(self.loc)
    #self.reverseCount[self.loc] += 1
    #self.reverseMemory[self.loc] = feat

    # Pop oldest feature from history, etc. if we hit buffer limit
    if len(self.history) > MEMORY_LENGTH:
      poppedFeat = self.history[0]
      del self.history[0]
      self.count[poppedFeat] -= 1


  @staticmethod
  def computeBasisTransform(locOffset1, motorOffset1, locOffset2, motorOffset2):
    locOffsetX1, locOffsetY1 = locOffset1
    motorOffsetX1, motorOffsetY1 = motorOffset1
    locOffsetX2, locOffsetY2 = locOffset2
    motorOffsetX2, motorOffsetY2 = motorOffset2

    # Solve system of equations to determine basis transform matrix
    coefficients = [
      [locOffsetX1, 0, locOffsetY1, 0],
      [0, locOffsetX1, 0, locOffsetY1],
      [locOffsetX2, 0, locOffsetY2, 0],
      [0, locOffsetX2, 0, locOffsetY2],
    ]
    ordinates = [motorOffsetX1, motorOffsetY1, motorOffsetX2, motorOffsetY2]
    try:
      basisValues = np.linalg.solve(coefficients, ordinates)
    except np.linalg.linalg.LinAlgError:
      return None
    basisTransform = basisValues.reshape((2, 2))
    assert basisValues[0] == basisTransform[0][0]
    assert basisValues[1] == basisTransform[0][1]
    assert basisValues[2] == basisTransform[1][0]
    assert basisValues[3] == basisTransform[1][1]

    return basisTransform


  def measureConsistency(self):
    best = 0.0
    bestTransform = None
    for m1 in self.motorMap.keys():
      for m2 in self.motorMap.keys():
        # Find the basis transform for these two location/world offset pairs
        locOffset1 = self.offsets[self.offsetMap[m1][0]]
        motorOffset1 = self.motorMap[m1]
        locOffset2 = self.offsets[self.offsetMap[m2][0]]
        motorOffset2 = self.motorMap[m2]
        basisTransform = self.computeBasisTransform(locOffset1, motorOffset1, locOffset2, motorOffset2)
        if basisTransform is None:
          continue

        # Check how many mappings match this transform
        n = 0
        consistent = 0
        for m in self.motorMap.keys():
          n += 1
          worldOffset = self.motorMap[m]
          locOffset = self.offsets[self.offsetMap[m][0]]
          transformed = np.dot(locOffset, basisTransform)
          transformed = [((v + 1) % self.locWidth) - 1 for v in transformed]
          if np.allclose(transformed, worldOffset):
            consistent += 1
        score = float(consistent) / float(n)
        if score > best:
          bestTransform = basisTransform
        best = max(best, score)

    return best, bestTransform



def testParams(locationWidth, worldWidth, initialWeight, inc, dec):
  iterationsToPerfect = []
  numTrials = 10
  for trial in xrange(numTrials):
    exp = Experiment(
      locationWidth=locationWidth,
      worldWidth=worldWidth,
      initialWeight=initialWeight,
      inc=inc,
      dec=dec,
    )
    for i in xrange(1000000):
      exp.runOne()
      if (i + 1) % 10000 == 0:
        consistency, basisT = exp.measureConsistency()
        if consistency > 0.99:
          iterationsToPerfect.append(i)
          break
    else:
      print "never got perfect"
      iterationsToPerfect.append(i)
  assert len(iterationsToPerfect) == numTrials
  total = 0.0
  for iterations in iterationsToPerfect:
    total += float(iterations)
  return total / float(numTrials)



if __name__ == "__main__":
  paramSet = [
    (3, 10, 0.05, 0.05, 0.01),
    (3, 10, 0.05, 0.05, 0.02),
  ]
  for params in paramSet:
    print "params: ", params
    averageIterationsToPerfect = testParams(*params)
    print "avg iterations: ", averageIterationsToPerfect
    print
