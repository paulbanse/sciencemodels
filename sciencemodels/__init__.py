"""sciencemodels: an agent-based model of science on epistemic landscapes.

Public API re-exports (the pre-restructure ``mesa_model`` /
``landscapefunctions`` interface):

- ``MyModel`` and ``Scientist``: the model and agent classes
- ``generate_data_parametric_exploration``: the sweep interface (API-stable
  per ADR-0002)
- the landscape generators (``randominit``, ``Gaussian``, ``noisyGaussian``,
  ``multipleGaussians``) and the agent-curiosity generators (``beta``,
  ``uniform``)

Submodules follow the conceptual levels: ``agents``, ``model``,
``landscapes``, ``metrics``, ``viz``, ``io``, ``plotting``.
"""
from .agents import Scientist
from .landscapes import (
    Gaussian,
    beta,
    multipleGaussians,
    noisyGaussian,
    randominit,
    uniform,
)
from .model import MyModel, generate_data_parametric_exploration

__all__ = [
    "Gaussian",
    "MyModel",
    "Scientist",
    "beta",
    "generate_data_parametric_exploration",
    "multipleGaussians",
    "noisyGaussian",
    "randominit",
    "uniform",
]
