from abstract_interpretation import domains
from abstract_interpretation.neural_network import *
from abstract_interpretation.verification import get_constraints
import numpy as np

net = NeuralNetwork([LinearLayer(16, 12), TanhLayer(), LinearLayer(12, 8), TanhLayer(), LinearLayer(8, 12), TanhLayer(), LinearLayer(12, 16)])

domain = domains.DeepPoly(lower_bounds = -np.ones(16, ), upper_bounds=np.ones(16, ))

get_constraints(net, domain)

