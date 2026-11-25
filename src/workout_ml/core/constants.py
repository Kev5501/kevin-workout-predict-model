"""Column names and data settings shared by training, inference and research."""
from __future__ import annotations

# Bad-data guard: a set is flagged when est_1rm / median(previous sessions) is outside this range
GUARD_RANGE = (0.5, 2.0)
GUARD_WINDOW = 5        # previous sessions of the same exercise + set
GUARD_MIN_SESSIONS = 3

# Feature-building settings (used by build_features and to rebuild features at inference)
FEATURE_SETTINGS = dict(
    time_lag_days=[90, 60, 30, 14, 7, 3, 1],
    metrics=['avg', 'max', 'min', 'slope', 'count', 'std'],
    binary_cols=[
        'injury_ind', 'illness_ind', 'girlfriend_ind', 'fatigue_ind', 'vacation_ind',
        'form_change_ind', 'forgot_headphones_ind', 'other_extenuating_circumstances_ind',
    ],
    binary_metrics=['avg', 'max'],
    ewma_alphas=[0.3, 0.5, 0.7],      # by session; est_1rm_ewma_30 / _50 / _70
    last_n=[2, 3],                    # average of the last n sessions
)

TARGET = 'est_1rm'
BASE_COL = 'est_1rm_most_recent'          # previous session of this exercise + set
EWMA_COL = 'est_1rm_ewma_50'
BASE_CHOICES = [BASE_COL, EWMA_COL]       # bases a model can predict a change from
FORCE_KEEP = [BASE_COL, EWMA_COL]
DAYS_SINCE_COL = 'days_since_last_exercise_set'
SERIES_COLS = ['exercise', 'set_number']
