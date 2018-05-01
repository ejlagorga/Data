import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.cluster import KMeans

def get_weights(X):
    N = X.shape[0]
    sigma_matrix = np.dot(X.T, X) / N
    u, s, v = np.linalg.svd(sigma_matrix)
    v = np.real(v)
 
    # Sort
    idx = s.argsort()[::-1]
    s = s[idx]
    u = u[:,idx]

    plt.plot(s)

    return s, u
 
def plot_relerr(X):
    l, v = get_weights(X)
    e = []
    
    for i in range(1, X.shape[1]):
        e.append(sum(np.square(l[:i]))/sum(np.square(l[:])))
    
    plt.plot(e)
    plt.title("Relative Error of PCA")
    plt.ylabel("relative error")
    plt.xlabel("principle components")
    plt.show()

def project_reduced_dim(X, dprime):
     l, v = get_weights(X)
     coefs = np.dot(X, v)
     reduced_X = np.dot(coefs[:, :dprime], v[:, :dprime].T)
     return reduced_X

def z_normalize(X):
    """
    Compute z-normalized data matrix.
    :param X: An n-by-d numpy array representing n points in R^d
    :return: An n-by-d numpy array representing n points in R^d that have been
    z-normalized.
    """
    mu = X.mean(0)
    sg = X.std(0)

    X = (X - mu) / sg
    return X


def show_2D_clusters(X, c):
    """
    Visualize the different clusters using color encoding.
    :param X: An n-by-d numpy array representing n points in R^d
    :param c: A list (or numpy array) of n elements. The ith entry, c[i], must
    be an integer representing the index of the cluster that point i belongs
    to.
    """
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=c)
    plt.show(block=False)  # this allows you to have multiple plots open


# Fetch the data from sklearn.
databunch = load_boston()

# Below, X is a n-by-d array of n points in R^d. feature_names is a list of
# strings; the jth entry identifying the features represented in the jth
# column of X.
X = databunch.data
# Normalize data
Xn = z_normalize(X)
feature_names = databunch.feature_names




# Plot relative error for each dimension
# plot_relerr(X)
# plot_relerr(Xn)
#
Xr2 = project_reduced_dim(X, 2)

# Do kmeans
nclusters = 5
kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(Xr2)
show_2D_clusters(Xr2, kmeans.labels_)

# Keep plots open
plt.show()



