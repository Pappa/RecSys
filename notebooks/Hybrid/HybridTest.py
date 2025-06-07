# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from recsys.MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator

import random
import numpy as np

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings)  = MovieLens.load()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#Simple RBM
SimpleRBM = RBMAlgorithm(epochs=40)
#Content
ContentKNN = ContentKNNAlgorithm()

#Combine them
Hybrid = HybridAlgorithm([SimpleRBM, ContentKNN], [0.5, 0.5])

evaluator.AddAlgorithm(SimpleRBM, "RBM")
evaluator.AddAlgorithm(ContentKNN, "ContentKNN")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
