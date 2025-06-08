# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from recsys.MovieLens import MovieLens
from recsys.algorithms.AutoRec import AutoRec
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
auto_rec = AutoRec()
evaluator.AddAlgorithm(auto_rec, "AutoRec")

# Just make random recommendations
random_rec = NormalPredictor()
evaluator.AddAlgorithm(random_rec, "Random")

# Fight!
evaluator.Evaluate(True, minimumRating=0.0)

evaluator.SampleTopNRecs(ml)
