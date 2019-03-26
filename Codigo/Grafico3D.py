import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC

# rs = np.random.RandomState(1234)

# Generate some fake data.
# n_samples = 200
# X is the input features by row.
# X = np.zeros((200,3))
# X[:n_samples/2] = rs.multivariate_normal( np.ones(3), np.eye(3), size=n_samples/2)
# X[n_samples/2:] = rs.multivariate_normal(-np.ones(3), np.eye(3), size=n_samples/2)

# Y is the class labels for each row of X.
# Y = np.zeros(n_samples); Y[n_samples/2:] = 1

# Fit the data with an svm
# svc = SVC(kernel='linear')
# svc.fit(X,Y)

# The equation of the separating plane is given by all x in R^3 such that:
# np.dot(svc.coef_[0], x) + b = 0. We should for the last coordinate to plot
# the plane in terms of x and y.
