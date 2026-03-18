"""
Backward-compatible re-export shim.

The analysis module has been split into three sub-modules:
  - app.features  — constants, feature engineering, data loading
  - app.models    — model fitting, prediction, ensemble blending
  - app.backtest  — hold-out tests, walk-forward CV, lead-time analysis

All public names are re-exported here so that existing imports
(e.g. `from app.analysis import fit_ensemble`) continue to work.
"""

from app.features import *   # noqa: F401,F403
from app.models import *     # noqa: F401,F403
from app.backtest import *   # noqa: F401,F403

# Explicit re-exports for private/underscore names used elsewhere
from app.features import _time_features, _uk_holidays  # noqa: F401
