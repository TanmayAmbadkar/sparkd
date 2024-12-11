from abstract_interpretation import domains
import numpy as np
import warnings
warnings.filterwarnings("ignore")
domain = domains.DeepPoly(lower_bounds=[-1, -1], upper_bounds=[1, 1])
# print("CONCRETE BOUNDS: ", domain.calculate_bounds())
new_dom = domain.affine_transform(W = np.array([[1.0, 1], [1.0, -1.0]]), b = np.array([0, 0]))
print("CONCRETE BOUNDS: ", new_dom.calculate_bounds())
new_dom = new_dom.relu()
print("CONCRETE BOUNDS: ", new_dom.calculate_bounds())
new_dom = new_dom.affine_transform(W = np.array([[1.0, 1], [1.0, -1.0]]), b = np.array([0, 0]))
print("CONCRETE BOUNDS", new_dom.calculate_bounds())
# new_dom = new_dom.relu()
# new_dom = new_dom.affine_transform(W = np.array([[1, 1], [0, 1]]), b = np.array([0, 0]))