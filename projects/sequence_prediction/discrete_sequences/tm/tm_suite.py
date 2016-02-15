#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
import json
import operator
import os
import random
import sys
import time

import numpy
from expsuite import PyExperimentSuite

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.research.monitor_mixin.trace import CountsTrace

from htmresearch.data.sequence_generator import SequenceGenerator
from htmresearch.support.sequence_prediction_dataset import ReberDataset
from htmresearch.support.sequence_prediction_dataset import SimpleDataset
from htmresearch.support.sequence_prediction_dataset import HighOrderDataset




# MIN_ORDER = 6
# MAX_ORDER = 7
# NUM_PREDICTIONS = [1, 2, 4]
# NUM_RANDOM = 1
# PERTURB_AFTER = 10000
# TEMPORAL_NOISE_AFTER = 5000

MIN_ORDER = 6
MAX_ORDER = 7
NUM_PREDICTIONS = [1]
NUM_RANDOM = 1

NUM_SYMBOLS = SequenceGenerator.numSymbols(MAX_ORDER, max(NUM_PREDICTIONS))
RANDOM_START = NUM_SYMBOLS
RANDOM_END = NUM_SYMBOLS + 5000

MODEL_PARAMS = {
  "model": "CLA",
  "version": 1,
  "predictAheadTime": None,
  "modelParams": {
    "inferenceType": "TemporalMultiStep",
    "sensorParams": {
      "verbosity" : 0,
      "encoders": {
        "element": {
          "fieldname": u"element",
          "name": u"element",
          "type": "SDRCategoryEncoder",
          "categoryList": range(max(RANDOM_END, NUM_SYMBOLS)),
          "n": 2048,
          "w": 41
        }
      },
      "sensorAutoReset" : None,
    },
      "spEnable": False,
      "spParams": {
        "spVerbosity" : 0,
        "globalInhibition": 1,
        "columnCount": 2048,
        "inputWidth": 0,
        "numActiveColumnsPerInhArea": 40,
        "seed": 1956,
        "columnDimensions": 0.5,
        "synPermConnected": 0.1,
        "synPermActiveInc": 0.1,
        "synPermInactiveDec": 0.01,
        "maxBoost": 0.0
    },
    "tpEnable" : True,
    "tpParams": {
      "verbosity": 0,
        "columnCount": 2048,
        "cellsPerColumn": 32,
        "inputWidth": 2048,
        "seed": 1960,
        "temporalImp": "monitored_tm_py",
        "newSynapseCount": 32,
        "maxSynapsesPerSegment": 128,
        "maxSegmentsPerCell": 128,
        "initialPerm": 0.21,
        "connectedPerm": 0.50,
        "permanenceInc": 0.1,
        "permanenceDec": 0.1,
        "predictedSegmentDecrement": 0.01,
        "globalDecay": 0.0,
        "maxAge": 0,
        "minThreshold": 15,
        "activationThreshold": 15,
        "outputType": "normal",
        "pamLength": 1,
      },
      "clParams": {
        "implementation": "cpp",
        "regionName" : "CLAClassifierRegion",
        "clVerbosity" : 0,
        "alpha": 0.0001,
        "steps": "1",
      },
      "trainSPNetOnlyIfRequested": False,
    },
}




def getEncoderMapping(model):
  encoder = model._getEncoder().encoders[0][1]
  mapping = dict()

  for i in range(NUM_SYMBOLS):
    mapping[i] = set(encoder.encode(i).nonzero()[0])

  return mapping



def classify(mapping, activeColumns, numPredictions):
  scores = [(len(encoding & activeColumns), i) for i, encoding in mapping.iteritems()]
  random.shuffle(scores)  # break ties randomly
  return [i for _, i in sorted(scores, reverse=True)[:numPredictions]]



class Suite(PyExperimentSuite):

  def reset(self, params, repetition):
    random.seed(params['seed'])

    if params['dataset'] == 'simple':
      self.dataset = SimpleDataset()
    elif params['dataset'] == 'reber':
      self.dataset = ReberDataset(maxLength=params['max_length'])
    elif params['dataset'] == 'high-order':
      self.dataset = HighOrderDataset(numPredictions=params['num_predictions'])
    else:
      raise Exception("Dataset not found")

    # if not os.path.exists(resultsDir):
    #   os.makedirs(resultsDir)
    # self.resultsFile = open(os.path.join(resultsDir, "0.log"), 'w')
    if params['verbosity'] > 0:
      print " initializing HTM model..."
    self.model = ModelFactory.create(MODEL_PARAMS)
    self.model.enableInference({"predictedField": "element"})
    self.shifter = InferenceShifter()
    self.mapping = getEncoderMapping(self.model)

    self.currentSequence = self.dataset.generateSequence()
    self.numPredictedActiveCells = []
    self.numPredictedInactiveCells = []
    self.numUnpredictedActiveColumns = []

    self.currentSequence = self.dataset.generateSequence()
    self.perturbed = False
    self.randoms = []
    self.verbosity = 1
    self.sequenceCounter = 0


  def replenish_sequence(self, params, iteration):
    if iteration > params['perturb_after'] and not self.perturbed:
      print "PERTURBING"
      sequence = self.dataset.generateSequence(perturbed=True)
      self.perturbed = True
    else:
      sequence = self.dataset.generateSequence()

    if iteration > params['inject_noise_after']:
      injectNoiseAt = random.randint(1, 3)
      print "injectNoiseAt: ", injectNoiseAt
      sequence[injectNoiseAt] = random.randrange(RANDOM_START, RANDOM_END)
      print sequence[injectNoiseAt]

    sequence.append(random.randrange(RANDOM_START, RANDOM_END))
    if params['verbosity'] > 0:
      print "Add sequence to buffer"
      print sequence
    self.currentSequence += sequence


  def iterate(self, params, repetition, iteration):
    element = self.currentSequence.pop(0)

    # whether there will be a random symbol after the current record
    randomFlag = (len(self.currentSequence) == 1)
    self.randoms.append(randomFlag)

    result = self.shifter.shift(self.model.run({"element": element}))
    tm = self.model._getTPRegion().getSelf()._tfdr

    tm.mmClearHistory()
    # Use custom classifier (uses predicted cells to make predictions)
    predictiveColumns = set([tm.columnForCell(cell) for cell in tm.predictiveCells])
    topPredictions = classify(self.mapping, predictiveColumns, params['num_predictions'])

    truth = None if (self.randoms[-1] or
                     len(self.randoms) >= 2 and self.randoms[-2]
                     ) else self.currentSequence[0]

    correct = None if truth is None else (truth in topPredictions)

    data = {"iteration": iteration,
            "current": element,
            "reset": False,
            "random": randomFlag,
            "train": True,
            "predictions": topPredictions,
            "truth": truth,
            "sequenceCounter": self.sequenceCounter}

    if params['verbosity'] > 0:
      print ("iteration: {0} \t"
             "current: {1} \t"
             "predictions: {2} \t"
             "truth: {3} \t"
             "correct: {4} \t").format(
        iteration, element, topPredictions, truth, correct)

    if len(self.currentSequence) == 0:
      self.replenish_sequence(params, iteration)
      self.sequenceCounter += 1

    return data


if __name__ == '__main__':
  suite = Suite()
  suite.start()
