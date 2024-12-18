from abstract_interpretation import domains
from abstract_interpretation.neural_network import *
from abstract_interpretation.verification import get_constraints
import numpy as np

domain = domains.DeepPoly(lower_bounds = -np.ones(2, ), upper_bounds=np.ones(2, ))

domain = domain.affine_transform(W = np.array([[1, 1], [1, -1]]), b = np.array([1, 1]))

domain = domain.relu()
domain = domain.affine_transform(W = np.array([[1, 1], [1, -1]]), b = np.array([1, 1]))

print(domain)
print(domain.calculate_bounds())