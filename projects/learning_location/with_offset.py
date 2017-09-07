# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""TODO"""

import collections
import csv

import numpy as np

ITERATIONS = 1000001
RESET_CHANCE = 0.0
RIGHT, LEFT, UP, DOWN, RESET = range(5)
ITERATIONS_PER_WORLD = 10000

FEATURE_SIZE = (25, 25)
NUM_FEATURES = 200

# Dimensions should be odd, middle element represents self-transition
OFFSET_WIDTH = 3
OFFSET_SIZE = 3 ** 2

LOC_WIDTH = 5
LOC_SIZE = LOC_WIDTH ** 2
INCREMENT = 1.0
DECREMENT = 1.0
BOOST = 0.00001
BOOST_DECAY = 0.0

MEMORY_LENGTH = 10000
REVERSE_MEM_LENGTH = 5

MOTOR_TXT = {
  RIGHT: "right",
  LEFT: "left",
  UP: "up",
  DOWN: "down",
}

STARTING_MOTOR = list(
    ([UP, DOWN] * 20) +
    ([LEFT, RIGHT] * 20) +
    ([UP, RIGHT, DOWN, LEFT]) * 20
)


def createTransitions():
  transitions = {}
  offsetCenter = int(OFFSET_WIDTH / 2)
  for i in xrange(LOC_WIDTH):
    for j in xrange(LOC_WIDTH):
      loc = (i * LOC_WIDTH) + j

      transitions[loc] = {}

      for oi in xrange(OFFSET_WIDTH):
        for oj in xrange(OFFSET_WIDTH):
          offset = (oi * OFFSET_WIDTH) + oj

          destX = ((oi - offsetCenter) + i) % LOC_WIDTH
          destY = ((oj - offsetCenter) + j) % LOC_WIDTH
          dest = (destX * LOC_WIDTH) + destY
          transitions[loc][offset] = dest
  return transitions


def getNextMotor(x, y):
  global STARTING_MOTOR
  if STARTING_MOTOR:
    motor = STARTING_MOTOR[0]
    del STARTING_MOTOR[0]
    return motor
  if np.random.random() < RESET_CHANCE:
    return RESET
  candidates = []
  if x > 0:
    candidates.append(LEFT)
  elif x < (FEATURE_SIZE[0] - 1):
    candidates.append(RIGHT)
  if y > 0:
    candidates.append(UP)
  elif y < (FEATURE_SIZE[1] - 1):
    candidates.append(DOWN)
  winner = np.random.randint(len(candidates))
  return candidates[winner]


def getWorld(numFeatures):
  return np.random.randint(numFeatures, size=np.prod(FEATURE_SIZE)).reshape(FEATURE_SIZE)


def currentCell(x, y):
  return (x * len(FEATURE_SIZE[0])) + y


def move(x, y, currentMotor):
  if currentMotor == LEFT:
    return (x - 1, y)
  elif currentMotor == RIGHT:
    return (x + 1, y)
  elif currentMotor == UP:
    return (x, y - 1)
  elif currentMotor == DOWN:
    return (x, y + 1)
  elif currentMotor == RESET:
    return (np.random.randint(FEATURE_SIZE[0]),
            np.random.randint(FEATURE_SIZE[1]))
  else:
    raise Exception('somethign went wrong')


def selectLocation(loc, motor, motorMap, transitions):
  if motor == RESET:
    return np.random.randint(LOC_SIZE)
  offset = motorMap[motor].argmax()
  return transitions[loc][offset]


def updateMotor(oldLoc, currentMotor, loc, motorMap, transitions):
  currentOffset = motorMap[currentMotor].argmax()
  #print "OLD: {} {}".format(currentMotor, currentOffset)
  for offset, dest in transitions[oldLoc].iteritems():
    if dest == loc:
      # Update the motor map so current motor maps to the correct offset from oldLoc to loc
      #print "NEW: {} {}".format(currentMotor, offset)
      #print
      adjustMotorWeights(motorMap, currentMotor, offset)
      break


def adjustMotorWeights(motorMap, currentMotor, newOffset):
  motorMap[currentMotor] -= DECREMENT
  motorMap[currentMotor][newOffset] += (INCREMENT + DECREMENT)
  motorMap[currentMotor] = motorMap[currentMotor].clip(0.0, 1.0)


def test(motorMap, transitions):
  total = 0
  correct = 0
  noBoosts = np.zeros((LOC_SIZE,), dtype=float)
  loc = 0
  for pair in [
      (LEFT, RIGHT),
      (RIGHT, LEFT),
      (UP, DOWN),
      (DOWN, UP)]:
    intermediate = selectLocation(loc, pair[0], motorMap, transitions)
    result = selectLocation(intermediate, pair[1], motorMap, transitions)
    total += 1
    if result == loc and loc != intermediate:
      correct += 1
  return float(correct) / float(total)


def main():
  featX = 12
  featY = 13
  loc = 0

  statsFile = open("stats.csv", "w")
  statsOut = csv.writer(statsFile)
  statsOut.writerow(("n", "no prediction", "bad cycle", "good cycle", "reciprocals"))

  # Map from feature to most recent location
  memory = {}
  # How many instances of each feature are in the history
  count = collections.defaultdict(int)
  # List of features in short term memory
  history = []

  reverseMemory = {}
  reverseHistory = []
  reverseCount = collections.defaultdict(int)

  stats = {
      "n": 0,
      "noPrediction": 0,
      "unknownPrediction": 0,
      "badPrediction": 0,
      "goodPrediction": 0,
  }

  transitions = createTransitions()
  #for i in transitions:
  #  for m in transitions[i]:
  #    print "{} {} {}".format(i, m, transitions[i][m])

  # Create mapping from motor command to offset
  motorMap = {}
  for m in (LEFT, RIGHT, UP, DOWN):
    motorMap[m] = np.zeros((OFFSET_SIZE,), dtype=float)
    motorMap[m][np.random.randint(OFFSET_SIZE)] = 0.1
  print motorMap

  # TODO: get rid of boosting
  boosts = np.zeros((LOC_SIZE,), dtype=float)
  nActive = np.zeros((LOC_SIZE,), dtype=int)

  for step in xrange(ITERATIONS):
    if (step % ITERATIONS_PER_WORLD) == 0:
      features = getWorld(NUM_FEATURES)
      # TODO: Do we need to do a reset or partial reset or is it ok for algo not to know?
      # TODO: For now, we make it easier for the algorithm by resetting the buffer
      history = []
      count = collections.defaultdict(int)
      reverseHistory = []
      reversecount = collections.defaultdict(int)
    if (stats["n"] % 10000) == 0:
      statsOut.writerow((stats["n"], stats["noPrediction"], stats["badPrediction"], stats["goodPrediction"], test(motorMap, transitions)))
    stats["n"] += 1

    currentMotor = getNextMotor(featX, featY)

    oldFeatX, oldFeatY = (featX, featY)
    featX, featY = move(featX, featY, currentMotor)

    oldLoc = loc
    loc = selectLocation(loc, currentMotor, motorMap, transitions)

    if currentMotor == RESET:
      history = []
      count = collections.defaultdict(int)
      reverseHistory = []
      reversecount = collections.defaultdict(int)
      continue

    feat = features[featX][featY]

    if count[feat] > 0:
      #print "cycle!"
      if loc == memory[feat]:
        stats["goodPrediction"] += 1
      else:
        stats["badPrediction"] += 1
      # TODO: Do we need to decrement only the current mistaken location or is it ok to keep decrementing all transitions for this motor command?
      loc = memory[feat]
      updateMotor(oldLoc, currentMotor, loc, motorMap, transitions)
    else:
      #print "no cycle!"
      if transitions[oldLoc][currentMotor]:
        stats["unknownPrediction"] += 1
      else:
        stats["noPrediction"] += 1

    if reverseCount[loc] > 0 and reverseMemory[loc] != feat:
      # Uh oh, we think we know the right location but it can't be right!
      # This motor-> offset isn't right so pick a random one
      adjustMotorWeights(motorMap, currentMotor, np.random.randint(OFFSET_SIZE))

    # Update boosting
    #boosts += BOOST
    #boosts[loc] *= BOOST_DECAY
    nActive[loc] += 1

    # Add new feature to history, etc
    history.append(feat)
    count[feat] += 1
    memory[feat] = loc

    reverseHistory.append(loc)
    reverseCount[loc] += 1
    reverseMemory[loc] = feat

    # Pop oldest feature from history, etc. if we hit buffer limit
    if len(history) > MEMORY_LENGTH:
      del history[0]
      count[feat] -= 1
    if len(reverseHistory) > REVERSE_MEM_LENGTH:
      del reverseHistory[0]
      reverseCount[loc] -= 1

  statsFile.close()
  print motorMap
  print "Duty cycles:"
  for v in np.sort(nActive):
    print float(v) / float(ITERATIONS)



if __name__ == "__main__":
  main()
