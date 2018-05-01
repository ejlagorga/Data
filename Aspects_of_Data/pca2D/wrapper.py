import numpy as np
import John_LaGorga_run as run

# Can change mean of Gaussian generating data
def get_some_data_matrix():
        X = np.random.multivariate_normal(mean=(1, -1), cov=((1, .2), (0.5, .3)),
                                                      size=30)
        return X

X = get_some_data_matrix()
run.plot_principle_components_2D(X)
run.plot_dim_reduced_data(X)
