"""Workout est_1rm prediction: data loading, features, model selection, inference.

Scripts, run from the ``research`` folder with the project venv::

    ..\\.venv\\Scripts\\python -m modeling.build_features            # Sheets -> data/features.parquet
    ..\\.venv\\Scripts\\python -m modeling.run_pipeline --quick      # smoke run
    ..\\.venv\\Scripts\\python -m modeling.run_pipeline --device auto
"""
