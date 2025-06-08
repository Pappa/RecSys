# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from recsys.MovieLens import MovieLens
from AutoRecAlgorithm import AutoRecAlgorithm
from surprise import NormalPredictor
from recsys.Evaluator import Evaluator
from EvaluatedAlgorithm import EvaluatedAlgorithm

import random
import numpy as np

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings)  = MovieLens.load()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#Autoencoder
auto_rec = AutoRecAlgorithm()
evaluator.AddAlgorithm(auto_rec, "AutoRec", algo_cls=EvaluatedAlgorithm)

# Just make random recommendations
random_rec = NormalPredictor()
evaluator.AddAlgorithm(random_rec, "Random")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
