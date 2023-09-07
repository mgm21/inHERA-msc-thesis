import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from IPython.display import clear_output
import functools
import time

import jax

# Uncomment these lines to use the cpu instead of the default gpu
jax.config.update('jax_platform_name', 'cpu')
print(jax.numpy.ones(3).device()) # TFRT_CPU_0

import jax.numpy as jnp

import brax
import jumanji
import qdax

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, compute_euclidean_centroids, MapElitesRepertoire
from qdax import environments
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.plotting import plot_map_elites_results
from qdax.utils.plotting import plot_multidimensional_map_elites_grid

from qdax.utils.metrics import CSVLogger, default_qd_metrics

from jax.flatten_util import ravel_pytree

from IPython.display import HTML
from brax.io import html

if "COLAB_TPU_ADDR" in os.environ:
  from jax.tools import colab_tpu
  colab_tpu.setup_tpu()

# clear_output()

from typing import Tuple, Any, Dict
import numpy

import jax
from functools import partial
from functools import partialmethod
import jax.numpy as jnp
from jax import jit, vmap, lax,make_jaxpr

import optax


import time
import json 

# Matteo's personal imports 
from jax.tree_util import tree_structure
from qdax.tasks.brax_envs import create_brax_scoring_fn
from datetime import datetime
from dataclasses import dataclass
import itertools
import matplotlib.pyplot as plt
import gc
import shutil
from importlib import import_module
import argparse
from distutils.dir_util import copy_tree
import seaborn as sns
import scipy

