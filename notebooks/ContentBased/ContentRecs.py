# -*- coding: utf-8 -*-
"""
Created on Fri May  4 16:25:39 2018

@author: Frank
"""


from recsys.MovieLens import MovieLens
from ContentKNNAlgorithm import ContentKNNAlgorithm
from Evaluator import Evaluator
from surprise import NormalPredictor

import random
import numpy as np

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings)  = MovieLens.load()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)


