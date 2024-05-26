"""Utility functions for Metaflow runs."""

from metaflow import Flow


def get_latest_successful_run(flow_name, tag):
    """Gets the latest successful run for a flow with a specific tag."""
    for run in Flow(flow_name).runs(tag):
        if run.successful:
            return run
    return None
