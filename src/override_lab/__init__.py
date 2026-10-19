"""Override Lab: a sandboxed model-organism study of alignment training and oversight.

The package is split so that everything a reviewer must trust without a GPU
(simulator, planner, parser, policy oracle, scenario generator, metrics,
statistics) imports only the Python standard library plus numpy/scipy.
Model-facing code (``models``, ``train``, parts of ``eval``) imports torch
lazily so the core can be tested on any machine.
"""

__version__ = "0.1.0"

# Bump whenever simulator semantics change. It is written into every frozen
# split manifest and every trace, so results produced under different rules
# can never be silently pooled.
SIMULATOR_VERSION = "meridian-1.0"
