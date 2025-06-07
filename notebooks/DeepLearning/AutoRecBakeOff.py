# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from recsys.MovieLens import MovieLens
from AutoRecAlgorithm import AutoRecAlgorithm
from surprise import NormalPredictor
from recsys.Evaluator import Evaluator

import random
import numpy as np

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings)  = MovieLens.load()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#Autoencoder
AutoRec = AutoRecAlgorithm()
evaluator.AddAlgorithm(AutoRec, "AutoRec")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
