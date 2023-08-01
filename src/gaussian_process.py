# This GP class was inspired by Rasmussen's GPs for ML book (mainly algo on p. 19) and Antoine Cully's unpublished implementation of GPs

from src.utils.all_imports import *
from jax.scipy import optimize

class GaussianProcess:
    def __init__(self, verbose=False):
        self.obs_noise = 0
        self.length_scale = 1
        self.rho = 0.4
        self.verbose = verbose

        # TODO: check why the sum below? Is it required in higher-dims?
        self.kernel = lambda x1, x2: jnp.exp(-jnp.sum((x1 - x2) ** 2 / (2*self.length_scale**2)))

        # From ITE paper:
        self.d = lambda x1, x2: jnp.linalg.norm(x2-x1)
        self.kernel = lambda x1, x2: (1 + ((jnp.sqrt(5)*self.d(x1, x2)) / (self.rho)) 
                                      + ((5*self.d(x1, x2)**2) / (3*self.rho**2))) * jnp.exp((-jnp.sqrt(5) * self.d(x1, x2))/(self.rho))

    # TODO: Would the following be interesting to me for this project? Not returning values of mu and var over the whole input space but by 
    # returning functional expressions for both and then can query from those if need be (and can create repertoires at the very end if need be)
    # TODO: Also, please note that here y_priors is the mean value of the prior function (nothing to do with ancestors, only with simu repertoire)
    # TODO: Instead of setting y_observed = y_observed - y_prior@x_observed as I do in ite.py, simply set y_observed = y_observed -y_prior@x_observed in this method at the top (so imports make more sense)
    def train(self,
              x_observed,
              y_observed,
              x_test,
              y_priors,):
        
        # TODO: change the following line to the private method call _get_K
        K = vmap(lambda x : vmap(lambda y: self.kernel(x, y))(x_observed))(x_observed) + self.obs_noise*jnp.eye(x_observed.shape[0])
        vec = vmap(lambda x: vmap(lambda y: self.kernel(x, y))(x_test))(x_observed)
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(jnp.transpose(L), jnp.linalg.solve(L, y_observed))

        # This line will have to be edited to sum in the adaptive prior
        mu = y_priors + jnp.matmul(jnp.transpose(vec), alpha)
        v = jnp.linalg.solve(L, vec)
        var = 1 - jnp.sum(jnp.square(v), axis=0)

        return mu, var
    
    # TODO: this could be made more efficient by inputting functions for mu and var as opposed to arrays and inputting a range over which the max is looked for
    def acquisition_function(self, mu, var, kappa=0.05):
        # kappa is a measure of how much uncertainty is valued
        # Should return the index of the policy to test given mu and var
        return jnp.argmax(mu + kappa*var)
    
    @partial(jit, static_argnums=(0,))
    def optimise_W(self, x_observed, y_observed, y_priors,):
        W0 = jnp.full(shape=len(y_priors), fill_value=1/len(x_observed))
        # print(f"W0: {W0}")

        K = self._get_K(x_observed=x_observed)
        # print(f"K: {K}")

        partial_likelihood = partial(self._get_likelihood, K=K, y_observed=y_observed, y_priors=y_priors)
        # print(f"partial_likelihood: {partial_likelihood}")

        # TODO: may not need partial above because optimize.minimize supports args tuple (see https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.optimize.minimize.html)
        opt_res = optimize.minimize(fun=partial_likelihood, x0=W0, method="BFGS")
        # print(f"opt_res: {opt_res}")

        W = opt_res.x
        # print(f"W: {W}")

        # Normalise W between [0, -1] # TODO: could change the way W is normalised
        W = W / jnp.sum(jnp.abs(W))
        # print(f"W: {W}")

        return W
    
    # TODO: This could made more efficient by using Cholesky decomposition method
    # TODO: Should this method be in here or in the InhertingAgent?
    @partial(jit, static_argnums=(0,))
    def _get_likelihood(self, W, K, y_observed, y_priors):
        A = y_observed - y_priors.T @ W
        llh = -0.5 * A.T @ jnp.linalg.inv(K) @ A - 0.5 * jnp.log(jnp.linalg.det(K))
        return -llh

    # TODO: this function can be removed if/when Cholesky method for likelihood is used
    def _get_K(self, x_observed):
        K = vmap(lambda x : vmap(lambda y: self.kernel(x, y))(x_observed))(x_observed) + self.obs_noise*jnp.eye(x_observed.shape[0])
        return K


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
                            [1, 0, 0]])
    
    y_observed = jnp.array([0, 2])

    # mu, var = gp.train(x_observed=x_observed,
    #                    y_observed=y_observed,
    #                    x_test=x_test,
    #                    y_priors=y_priors)


    # Prior values of the ancestors at the observed points
    # The last one is exactly the y_observed and should "explain" our current points the best
    y_priors = jnp.array([[0, 1], [1, 0], [1, 0], [0, 2]])
    print(gp.optimise_W(x_observed=x_observed, y_observed=y_observed, y_priors=y_priors))

    # # Testing the _get_likelihood method alone
    # # Check that it works by varying the parameters here and observing that giving the last weight all the importance
    # # should yield the lowest neg llh
    # W = jnp.array([0.09, 0.01, 0, 0.9])
    # K = gp._get_K(x_observed)
    # print(gp._get_likelihood(W, K, y_observed, y_priors))
    

    

    






