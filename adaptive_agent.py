from all_imports import *

class AdaptiveAgent:
    def __init__(self,
                 name="some_name",
                 x_observed=jnp.array([]),
                 y_observed=jnp.array([]),
                 mu=jnp.array([]),
                 var=jnp.array([])):
        
        self.name = name
        self.x_observed = x_observed
        self.y_observed = y_observed
        self.mu = mu
        self.var = var
    

