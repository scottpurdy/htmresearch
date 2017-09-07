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

ITERATIONS = 50000001
ITERATIONS_PER_WORLD = 10000
FEATURE_SIZE = (25, 25)
NUM_FEATURES = 200
LAYER_SIZE = 25
MEMORY_LENGTH = 10000
RIGHT, LEFT, UP, DOWN, RESET = range(5)
RESET_CHANCE = 0.0
LOC_SIZE = 25
INCREMENT = 1.0
DECREMENT = 0.03
BOOST = 0.00001
BOOST_DECAY = 0.0


def getNextMotor(x, y):
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


def selectLocation(loc, motor, transitions, boosts):
  if motor == RESET:
    return np.random.randint(LOC_SIZE)
  weights = np.copy(boosts)
  for cell, perm in transitions[loc][motor].iteritems():
    try:
      weights[cell] += perm
    except:
      print cell
      print perm
      print weights
      raise
  return weights.argmax()


def updateTransitions(oldLoc, motor, loc, transitions):
  keys = transitions[oldLoc][motor].keys()
  for k in keys:
    if k != loc:
      #transitions[oldLoc][motor][k] -= DECREMENT
      transitions[oldLoc][motor][k] = 0
      if transitions[oldLoc][motor][k] <= 0.0:
        del transitions[oldLoc][motor][k]
  transitions[oldLoc][motor][loc] += INCREMENT
  if transitions[oldLoc][motor][loc] > 1.0:
    transitions[oldLoc][motor][loc] = 1.0


def test(transitions):
  total = 0
  correct = 0
  noBoosts = np.zeros((LOC_SIZE,), dtype=float)
  for loc in transitions:
    for pair in [
        (LEFT, RIGHT),
        (RIGHT, LEFT),
        (UP, DOWN),
        (DOWN, UP)]:
      intermediate = selectLocation(loc, pair[0], transitions, noBoosts)
      result = selectLocation(intermediate, pair[1], transitions, noBoosts)
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

  stats = {
      "n": 0,
      "noPrediction": 0,
      "unknownPrediction": 0,
      "badPrediction": 0,
      "goodPrediction": 0,
  }

  # Maps cell to motor to new cell to permanence
  transitions = {}
  for i in xrange(LOC_SIZE):
    transitions[i] = {}
    for j in xrange(4):
      transitions[i][j] = collections.defaultdict(float)

  boosts = np.zeros((LOC_SIZE,), dtype=float)
  nActive = np.zeros((LOC_SIZE,), dtype=int)


  for step in xrange(ITERATIONS):
    if (step % ITERATIONS_PER_WORLD) == 0:
      features = getWorld(NUM_FEATURES)
      # TODO: Do we need to do a reset or partial reset or is it ok for algo not to know?
      # TODO: For now, we make it easier for the algorithm by resetting the buffer
      history = []
      count = collections.defaultdict(int)
    if (stats["n"] % 10000) == 0:
      statsOut.writerow((stats["n"], stats["noPrediction"], stats["badPrediction"], stats["goodPrediction"], test(transitions)))
    stats["n"] += 1

    currentMotor = getNextMotor(featX, featY)

    oldFeatX, oldFeatY = (featX, featY)
    featX, featY = move(featX, featY, currentMotor)

    oldLoc = loc
    loc = selectLocation(loc, currentMotor, transitions, boosts)

    if currentMotor == RESET:
      history = []
      count = collections.defaultdict(int)
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
      updateTransitions(oldLoc, currentMotor, loc, transitions)
    else:
      #print "no cycle!"
      if transitions[oldLoc][currentMotor]:
        stats["unknownPrediction"] += 1
      else:
        stats["noPrediction"] += 1

    # Update boosting
    boosts += BOOST
    boosts[loc] *= BOOST_DECAY
    nActive[loc] += 1

    # Add new feature to history, etc
    history.append(feat)
    count[feat] += 1
    memory[feat] = loc

    # Pop oldest feature from history, etc. if we hit buffer limit
    if len(history) > MEMORY_LENGTH:
      del history[0]
      count[feat] -= 1
      # Commented out because it shouldn't matter if this is left around since we have the count
      #if count[feat] == 0:
      #  del memory[feat]

  statsFile.close()
  print "Duty cycles:"
  for v in np.sort(nActive):
    print float(v) / float(ITERATIONS)

  print "Best transitions:"
  for i in xrange(LOC_SIZE):
    print
    print "Loc", i
    print "r", sorted(transitions[i][RIGHT].iteritems(), key=lambda x: x[1])[-1][0]
    print "d", sorted(transitions[i][DOWN].iteritems(), key=lambda x: x[1])[-1][0]
    print "l", sorted(transitions[i][LEFT].iteritems(), key=lambda x: x[1])[-1][0]
    print "u", sorted(transitions[i][UP].iteritems(), key=lambda x: x[1])[-1][0]



if __name__ == "__main__":
  main()
