import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

import brew
from brew.base import Ensemble
from brew.combination.combiner import Combiner
from brew.stacking.stacker import EnsembleStack
from brew.stacking.stacker import EnsembleStackClassifier

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

layer_1 = [SVC(probability=True),
           RandomForestClassifier(n_estimators=100),
           ExtraTreesClassifier(n_estimators=100)]

layer_2 = [SVC(probability=True), LogisticRegression(max_iter=500)]

stack = EnsembleStack(cv=10) # number of folds per layer
stack.add_layer(Ensemble(layer_1))
stack.add_layer(Ensemble(layer_2))

clf = EnsembleStackClassifier(stack, Combiner('mean'))
