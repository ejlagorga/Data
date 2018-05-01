import matplotlib.pyplot as plt
import numpy as np
import random
from faces import get_faces
from faces import show_image_subset_reshaped

def get_weights(X):
    N = X.shape[0]
    sigma_matrix = np.dot(X.T, X) / N
    u, s, v = np.linalg.svd(sigma_matrix)
    v = np.real(v)

    # Sort
    idx = s.argsort()[::-1]   
    s = s[idx]
    u = u[:,idx]
    
    return s, u

def project_reduced_dim(X, v, dprime):
    coefs = np.dot(X, v)
    reduced_X = np.dot(coefs[:, :dprime], v[:, :dprime].T)
    return reduced_X


if __name__ == '__main__':
    X = get_faces()
    l, v = get_weights(X)
    e = []

    for i in range(1, 4096):
        e.append(sum(np.square(l[:i]))/sum(np.square(l[:])))

    plt.plot(e)
    plt.title("Relative Error of PCA")
    plt.ylabel("relative error")
    plt.xlabel("principle components")
    plt.show()

    Xr15 = project_reduced_dim(X, v, 15)
    show_image_subset_reshaped(Xr15)

    Xr100 = project_reduced_dim(X, v, 100)
    show_image_subset_reshaped(Xr100)

      
    

