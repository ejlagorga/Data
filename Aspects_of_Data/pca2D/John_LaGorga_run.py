import numpy as np
import matplotlib.pyplot as plt

def plotvec(m, v, **kwargs):
    """ Plot a vector in the plane.
    :param v: a shape (2,) or (2,1) numpy array
    """
    plt.plot([m[0], v[0]+m[0]], [m[1], v[1]+m[1]], **kwargs)

def plotfig(X, Vs):
    """ Plot a vector in the plane.
    :param X: Data
          Vs: List of Vectors
    """
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    m = np.mean(X, axis=0)
    for v in Vs:
        plotvec(m, v)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

def plot_principle_components_2D(X):
    """ Given a two dimensional array of numbers, X, where each row represents
    an observation, plot the observations in R^2, and overlay the principle
    directions.
    :param X: A 2D numpy array
    """

    cov = np.cov(X.T, ddof=1)
    [w,v] = np.linalg.eig(cov)
    Vs = [w[0]*v[:,0], w[1]*v[:,1]]
    plotfig(X, Vs)
    
def plot_dim_reduced_data(X):
    """ Plots each point in X projected on to the principle component.
    :param X: A 2D numpy array
    """

    cov = np.cov(X.T, ddof=1)
    [w,v] = np.linalg.eig(cov)
    i = w.argsort()[::-1][0]
    P = np.column_stack((np.dot(X, v[:,i]), np.zeros(np.shape(X)[0])))
    plotfig(P, [])
	

if __name__ == "__main__":
    # Below is boilerplate code to get your up and running. The particular
    # value of X will be changed when your code is exercised.
    X = np.random.multivariate_normal(mean=(-2, 2), cov=((1, 2), (0.5, 3)),
                                      size=100)
    plot_principle_components_2D(X)
    plot_dim_reduced_2D(X)



