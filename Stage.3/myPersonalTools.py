# my personal tools by python

import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter

def myMinMaxScale(data):
    dataContainer = (data - np.min(data))/(np.max(data)-np.min(data))
    return dataContainer

def myMeanScale(data):
    dataContainer = data/np.mean(data)
    return dataContainer

def mySigmoid(data):
    dataContainer = []
    for i in range(len(data)):
        dataContainer.append(1/(1+np.exp(data[i]-np.mean(data))))
    return dataContainer