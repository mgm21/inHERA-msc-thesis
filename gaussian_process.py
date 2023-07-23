# This GP class was inspired by Rasmussen's GPs for ML book (mainly algo on p. 19) and Antoine Cully's unpublished implementation of GPs

from all_imports import *

class GaussianProcess:
    def __init__(self):
        self.obs_noise = 0
        self.length_scale = 1

        # TODO: check why the sum below? Is it required in higher-dims?
        self.kernel = lambda x1, x2: jnp.exp(-jnp.sum((x1 - x2) ** 2 / (2*self.length_scale**2)))

    def train(self,
              x_observed,
              y_observed,
              x_test):
        
        K = vmap(lambda x : vmap(lambda y: self.kernel(x, y))(x_observed))(x_observed) + self.obs_noise*jnp.eye(x_observed.shape[0])
        vec = vmap(lambda x: vmap(lambda y: self.kernel(x, y))(x_test))(x_observed)
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(jnp.transpose(L), jnp.linalg.solve(L, y_observed))
        # This line will have to be edited to sum in the adaptive prior
        mu = jnp.matmul(jnp.transpose(vec), alpha)
        v = jnp.linalg.solve(L, vec)
        std = 1 - jnp.sum(jnp.square(v), axis=0)

        return mu, std


if __name__ == "__main__":
    gp = GaussianProcess()

    # Toy variables to debug the GP
    
    # We put all our discretised points in x_test (3 dims with 2 bins in each)
    x_test = jnp.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1],])
    
    x_observed = jnp.array([[0, 0, 0],
                            [1, 1, 1]])
    
    y_observed = jnp.array([0, 100])

    results = gp.train(x_observed=x_observed,
                       y_observed=y_observed,
                       x_test=x_test)
    
    print(results)
    
    # To check that kernel function is working appropriately
    # TODO: put unit tests throughout the code base later
    # print(gp.kernel(x1=jnp.array([0, 0, 0]),
    #                 x2=jnp.array([1, 1, 1])))

    # TODO: note that I'm getting a "v: jnp.DeviceArray" output and I'm not sure why

    # TODO: in the dummy example, I am not getting the value of 0 where I observed 0, and the uncertainty is not 0 where I observed 0, why is that?
    # TODO: weirdly, when I switch up the y_observed to 0, 100 instead of 100, 0, then it is true that the 0 is still not a definite 0, but the uncertainty
    #  is now not 0 for the 100 valued one (so the pattern is like so: the second observation never has a 0 uncertainty and the 0 value-d observation is never
    #  exactly 0 in the new mu; maybe this is due to the cholesky method or to jax. Anyhow the values are veruy close so it's probably not a problem)







