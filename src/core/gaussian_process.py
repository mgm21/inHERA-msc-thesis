# This GP class was inspired by Rasmussen's GPs for ML book (mainly algo on p. 19) and Antoine Cully's unpublished implementation of GPs

from src.utils.all_imports import *
import jaxopt
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative
from jaxopt.projection import projection_box
from jaxopt import ScipyBoundedMinimize

class GaussianProcess:
    def __init__(self, verbose=False, obs_noise=0.01, length_scale=1, rho=0.4, kappa=0.05, l1_regularisation_weight=0.001, invmax_regularisation_weight=0.0001, uncertainty_regularisation_weight=0.0001):
        self.obs_noise = obs_noise
        self.length_scale = length_scale
        self.rho = rho
        self.verbose = verbose
        self.kappa = kappa
        self.l1_regularisation_weight = l1_regularisation_weight
        self.invmax_regularisation_weight = invmax_regularisation_weight
        self.uncertainty_regularisation_weight = uncertainty_regularisation_weight
        self.set_default_gradient_projection_settings()

    def train(self, x_observed, y_observed, x_test, y_priors,):
        """GP training method using Cholesky decomposition presented on p.19 of Rasmussen book"""
        K = vmap(lambda x : vmap(lambda y: self.kernel(x, y))(x_observed))(x_observed) + self.obs_noise*jnp.eye(x_observed.shape[0])
        vec = vmap(lambda x: vmap(lambda y: self.kernel(x, y))(x_test))(x_observed)
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(jnp.transpose(L), jnp.linalg.solve(L, y_observed))

        # The following line was edited to sum in the adaptive prior
        mu = y_priors + jnp.matmul(jnp.transpose(vec), alpha)
        v = jnp.linalg.solve(L, vec)
        var = 1 - jnp.sum(jnp.square(v), axis=0)
        return mu, var
    
    def _train(self, x_observed, y_observed, x_test, y_priors,):
        """Naive GP training method which does not use Cholesky decomposition to make sure that GP is implemented correctly"""
        K = self._get_K(x_observed)
        k = vmap(lambda x: vmap(lambda y: self.kernel(x, y))(x_test))(x_observed)
        mu = y_priors + k.T @ jnp.linalg.inv(K) @ y_observed
        var = vmap(lambda test_point: self.kernel(test_point, test_point).T - (vmap(lambda x_obs: self.kernel(test_point, x_obs))(x_observed)).T @ jnp.linalg.inv(K) @ vmap(lambda x_obs: self.kernel(test_point, x_obs))(x_observed))(x_test)
        return mu, var

    def kernel(self, x1, x2):
        d = lambda x1, x2: jnp.linalg.norm(x2-x1)
        return (1 + ((jnp.sqrt(5)*d(x1, x2)) / (self.rho)) + ((5*d(x1, x2)**2) / (3*self.rho**2))) * jnp.exp((-jnp.sqrt(5) * d(x1, x2))/(self.rho))
    
    @partial(jit, static_argnums=(0,))
    # TODO: check with Antoine that the following is correct
    def acquisition_function(self, mu, var):
        return jnp.nanargmax(mu + self.kappa*jnp.sqrt(var))
    
    @partial(jit, static_argnums=(0,))
    def optimise_W(self, x_observed, y_observed, y_priors, y_priors_vars=None):
        W0 = jnp.full(shape=len(y_priors), fill_value=1/len(x_observed))
        K = self._get_K(x_observed=x_observed)
        partial_likelihood = partial(self.loss, K=K, y_observed=y_observed, y_priors=y_priors, y_priors_vars=y_priors_vars)
        pg = ProjectedGradient(fun=partial_likelihood, projection=self._gradient_projection_type)
        pg_sol = pg.run(W0, hyperparams_proj=self._projection_hyperparameters).params
        W = pg_sol
        return W
    
    @partial(jit, static_argnums=(0,))
    def _get_likelihood(self, W, K, y_observed, y_priors):
        A = y_observed - y_priors.T @ W
        llh = -0.5 * A.T @ jnp.linalg.inv(K) @ A - 0.5 * jnp.log(jnp.linalg.det(K))
        return -llh
    
    @partial(jit, static_argnums=(0,))
    def loss(self, W, K, y_observed, y_priors, y_priors_vars=None):
        """Return the unaltered likelihood"""
        loss = self._get_likelihood(W, K, y_observed, y_priors)
        return loss
    
    @partial(jit, static_argnums=(0,))
    def loss_regularised_l1(self, W, K, y_observed, y_priors, y_priors_vars=None):
        """Return the L1 regularised likelihood"""
        loss = self._get_likelihood(W, K, y_observed, y_priors) + self.l1_regularisation_weight * (jnp.sum(jnp.abs(W)))
        return loss
    
    @partial(jit, static_argnums=(0,))
    def loss_regularised_invmax(self, W, K, y_observed, y_priors, y_priors_vars=None):
        """Return the likelihood regularised with the inverse of the max of the weights"""
        loss = self._get_likelihood(W, K, y_observed, y_priors) + self.invmax_regularisation_weight * (1/jnp.max(W))
        return loss
    
    @partial(jit, static_argnums=(0,))
    def loss_regularised_l1_invmax(self, W, K, y_observed, y_priors, y_priors_vars=None):
        loss = self._get_likelihood(W, K, y_observed, y_priors) + self.l1_regularisation_weight * (jnp.sum(jnp.abs(W))) + self.invmax_regularisation_weight * (1/jnp.max(W))
        return loss
    
    @partial(jit, static_argnums=(0,))
    def loss_regularised_l1_invmax_uncertainty(self, W, K, y_observed, y_priors, y_priors_vars):
        uncertainty_loss = jnp.sum(W * jnp.sum(y_priors_vars, axis=1), axis=0) / len(y_observed)
        loss = self._get_likelihood(W, K, y_observed, y_priors) + self.l1_regularisation_weight * (jnp.sum(jnp.abs(W))) + self.invmax_regularisation_weight * (1/jnp.max(W)) + self.uncertainty_regularisation_weight * uncertainty_loss
        return loss
    
    @partial(jit, static_argnums=(0,))
    def _get_K(self, x_observed):
        K = vmap(lambda x : vmap(lambda y: self.kernel(x, y))(x_observed))(x_observed) + self.obs_noise*jnp.eye(x_observed.shape[0])
        return K

    def set_default_gradient_projection_settings(self,):
        self._gradient_projection_type = projection_box
        self._projection_hyperparameters = (-jnp.inf, jnp.inf)
    
    def set_box_gradient_projection(self,):
        self._gradient_projection_type = projection_box
    
    def set_non_negative_gradient_projection(self,):
        self._gradient_projection_type = projection_non_negative
    
    def set_projection_hyperparameters(self, hyperparams):
        self._projection_hyperparameters = hyperparams


if __name__ == "__main__":
    # # Debugging original un-regularised likelihood not working for GPCF
    # gp = GaussianProcess()
    # x_observed = jnp.array([[0.5933333,0.23333333, 0.29333332, 0.46, 0.56, 0.20666666]])
    # y_observed = jnp.array([0.29857272])
    # ancestor_mus_at_curr_obs = jnp.array([[0.3464623], [0.28252518], [0.29259598], [0.30373174], [0.24637341], [0.40413338]])
    # print(gp.optimise_W(x_observed=x_observed, y_observed=y_observed, y_priors=ancestor_mus_at_curr_obs))


    # Debugging 2: How does the following make any sense? It literally observed 40, shouldn't it predict that the ancestor is the one at 40?
    gp = GaussianProcess()
    gp.set_box_gradient_projection()
    gp.set_projection_hyperparameters(hyperparams=(-jnp.inf, +jnp.inf))
    gp.loss = gp.loss_regularised_l1_invmax_uncertainty

    x_observed = jnp.array([[0.5933333,0.23333333, 0.29333332, 0.46, 0.56, 0.20666666]])
    y_observed = jnp.array([0.4004079])
    ancestor_mus_at_curr_obs = jnp.array([[0.3464623], [0.28252518], [0.29259598], [0.30373174], [0.24637341], [0.40413338], [1.0308355]])
    ancestor_vars_at_curr_obs = jnp.array([[0.0], [0.0], [0.0], [0.0], [1.], [0.], [0.0]])
    K = gp._get_K(x_observed)

    weights_output_from_optim = gp.optimise_W(x_observed=x_observed, y_observed=y_observed, y_priors=ancestor_mus_at_curr_obs, y_priors_vars=ancestor_vars_at_curr_obs)
    weights_expected = jnp.array([0, 0, 0, 0, 0, 1, 0])

    print(f"weights_outputted: {weights_output_from_optim}") 
    print(f"weights_expected: {weights_expected}")
    print(f"loss from output of optim: {gp.loss(weights_output_from_optim, K, y_observed, ancestor_mus_at_curr_obs, ancestor_vars_at_curr_obs)}")
    print(f"loss from expected weights: {gp.loss(weights_expected, K, y_observed, ancestor_mus_at_curr_obs, ancestor_vars_at_curr_obs)}")
    print(f"likelihood from output of optim: {gp._get_likelihood(weights_output_from_optim, K, y_observed, ancestor_mus_at_curr_obs)}")
    print(f"likelihood from expected weights: {gp._get_likelihood(weights_expected, K, y_observed, ancestor_mus_at_curr_obs)}")

    # Debugging (rest)
    # # Following toy problem to make sure that the GP works as expected
    # gp = GaussianProcess()

    # # Toy variables to debug the GP
    
    # # We put all our discretised points in x_test (3 dims with 2 bins in each)
    # x_test = jnp.array([[0, 0, 0],
    #                     [0, 0, 1],
    #                     [0, 1, 0],
    #                     [0, 1, 1],
    #                     [1, 0, 0],
    #                     [1, 0, 1],
    #                     [1, 1, 0],
    #                     [1, 1, 1],])
    
    # x_observed = jnp.array([[0, 0, 0],
    #                         [1, 0, 0]])
    
    # y_observed = jnp.array([0, 2])

    # y_prior = jnp.zeros(shape=x_test.shape[0])

    # mu, var = gp.train(x_observed=x_observed,
    #                    y_observed=y_observed,
    #                    x_test=x_test,
    #                    y_priors=y_prior)
    
    # mu2, var2 = gp._train(x_observed=x_observed,
    #                    y_observed=y_observed,
    #                    x_test=x_test,
    #                    y_priors=y_prior)
    
    # print(f"mu: {mu}")
    # print(f"mu2: {mu2}")
    # print(f"var: {var}")
    # print(f"var2: {var2}")
    
    # x_observed = jnp.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    # y_observed = jnp.array([0, 2, 5, 6])

    # mu, var = gp.train(x_observed=x_observed,
    #                    y_observed=y_observed,
    #                    x_test=x_test,
    #                    y_priors=y_prior)
    
    # mu2, var2 = gp._train(x_observed=x_observed,
    #                    y_observed=y_observed,
    #                    x_test=x_test,
    #                    y_priors=y_prior)
    
    # print(f"mu: {mu}")
    # print(f"mu2: {mu2}")
    # print(f"var: {var}")
    # print(f"var2: {var2}")

    # # Prior values of the ancestors at the observed points
    # # The last one is exactly the y_observed and should "explain" our current points the best
    # y_priors = jnp.array([[0, 1, 3, 4], [1, 0, 2, 1], [1, 0, 1, 2], [0, 2, 5, 6]])
    # print("result of likelihood optimisation:")
    # print(gp.optimise_W(x_observed=x_observed, y_observed=y_observed, y_priors=y_priors))

    # # # Tested the _get_likelihood method alone
    # # # Check that it works by varying the parameters here and observing that giving the last weight all the importance
    # # # should yield the lowest neg llh
    # W = jnp.array([0, 0.0, 1, 0.1])
    # K = gp._get_K(x_observed)
    # print("result of calculating an arbitrary likelihood:")
    # print(gp._get_likelihood(W, K, y_observed, y_priors))


######### The below are all the TODO and remarks that were previously in the code above

"""

# TODO: potential problem with the GP:
# Get negative values for the variance sometimes,
# Cannot input the same input twice.

# Best might be to re-write the GP train method in the more traditional way and see the difference in results on this toy problem
# Especially to see if it solves some of the problems above.

# TODO: if the values of mu are super high, will variance ever really be taken into account?
# Will the GP not only ever go for the highest predicted mu which would then make the whole uncertainty component irrelevant
# And could lead to being stuck in an undesirable optimum.
# Therefore must either scale the mus or must scale the variance up to be on the same scale. THIS IS THE PROBLEM.

# from jax.scipy import optimize
# from scipy import optimize as scipy_optimize

# 06/08/2023 Trying different optimisation libraries

# TODO: check why the sum in the exponential kernel function for high-dimensions?

# About the train method:
# TODO: Would the following be interesting to me for this project? Not returning values of mu and var over the whole input space but by 
# returning functional expressions for both and then can query from those if need be (and can create repertoires at the very end if need be)
# TODO: Also, please note that here y_priors is the mean value of the prior function (nothing to do with ancestors, only with simu repertoire)
# TODO: Instead of setting y_observed = y_observed - y_prior@x_observed as I do in ite.py, simply set y_observed = y_observed -y_prior@x_observed in this method at the top (so imports make more sense)
# TODO: It could be confusing that both this method's y_priors and optimise_W's y_priors have the same name when they refer to different objects.
#  y_priors in train refers to the value of the simulated repertoire at x_observed
#  y_priors in optimise_W refers to the value of the ancestor mus at x_observed put in a matrix
# TODO: SET ONE OF THE TWO FOLLOWING ONES TO train FOT THAT TO BE THE ONE THAT IS USED
# TODO: change the K= line to the private method call _get_K

# Concerning acquisition function
# TODO: this could be made more efficient by inputting functions for mu and var as opposed to arrays and inputting a range over which the max is looked for

# From the optimise_W method:
 # # TODO: may not need partial above because optimize.minimize supports args tuple (see https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.optimize.minimize.html)
# TODO: the below is to use jax.scipy.optimize.minimize
# opt_res = optimize.minimize(fun=partial_likelihood, x0=W0, method="BFGS",)
# W = opt_res.x
# # TODO: the below is to transform into one-hot encoding
# idx_max = jnp.nanargmax(W)
# W = jnp.zeros(W.shape)
# W = W.at[idx_max].set(1)

# About _get_likelihood method:
# TODO: This could made more efficient by using Cholesky decomposition method
# TODO: Should this method be in here or in the InhertingAgent?

# If you want to use the exponential kernel
self.kernel = lambda x1, x2: jnp.exp(-jnp.sum((x1 - x2) ** 2 / (2*self.length_scale**2)))

"""




