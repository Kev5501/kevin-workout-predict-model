"""Errors shared by the API and the jobs."""


class Busy(RuntimeError):
    """Another run of the same kind is in progress."""
