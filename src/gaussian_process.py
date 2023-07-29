# This GP class was inspired by Rasmussen's GPs for ML book (mainly algo on p. 19) and Antoine Cully's unpublished implementation of GPs

from src.utils.all_imports import *

class GaussianProcess:
    def __init__(self):
        self.obs_noise = 0
        self.length_scale = 1
        self.rho = 0.4

        # TODO: check why the sum below? Is it required in higher-dims?
        self.kernel = lambda x1, x2: jnp.exp(-jnp.sum((x1 - x2) ** 2 / (2*self.length_scale**2)))

        # From ITE paper:
        self.d = lambda x1, x2: jnp.linalg.norm(x2-x1)
        self.kernel = lambda x1, x2: (1 + ((jnp.sqrt(5)*self.d(x1, x2)) / (self.rho)) 
                                      + ((5*self.d(x1, x2)**2) / (3*self.rho**2))) * jnp.exp((-jnp.sqrt(5) * self.d(x1, x2))/(self.rho))

    def train(self,
              x_observed,
              y_observed,
              x_test,
              y_prior,):
        
        K = vmap(lambda x : vmap(lambda y: self.kernel(x, y))(x_observed))(x_observed) + self.obs_noise*jnp.eye(x_observed.shape[0])
        vec = vmap(lambda x: vmap(lambda y: self.kernel(x, y))(x_test))(x_observed)
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(jnp.transpose(L), jnp.linalg.solve(L, y_observed))

        # This line will have to be edited to sum in the adaptive prior
        mu = y_prior + jnp.matmul(jnp.transpose(vec), alpha)
        v = jnp.linalg.solve(L, vec)
        var = 1 - jnp.sum(jnp.square(v), axis=0)

        return mu, var
    
    def acquisition_function(self, mu, var, kappa=0.05):
        # kappa is a measure of how much uncertainty is valued
        # Should return the index of the policy to test given mu and var
        return jnp.argmax(mu + kappa*var)


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
                            [0, 0, 0]])
    
    y_observed = jnp.array([0, 2])

    y_prior = jnp.zeros(shape=x_test.shape)
    print(y_prior)

    mu, var = gp.train(x_observed=x_observed,
                       y_observed=y_observed,
                       x_test=x_test,
                       y_prior=y_prior)
    






