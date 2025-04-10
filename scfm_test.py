import numpy as np
import sys
from scipy.optimize import *
from CompressibleFlowFunctions.misc import *

mdot= 136
Rs  = 296.38 #296.38
G   = 0.912606342 # 0.912606342

scfh = mdot_to_scfh(mdot,Rs,G)
