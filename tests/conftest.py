# Copyright (c) 2025 Microsoft Corporation.
"""Shared pytest fixtures for all test modules."""

import os

import pytest


@pytest.fixture(autouse=True)
def restore_cwd():
    """Restore the working directory after every test.

    load_config uses set_cwd=True by default, which calls os.chdir() to the
    config file's directory.  Without this fixture each test leaves the process
    in a (soon-to-be-deleted) tmp_path, causing os.getcwd() to raise
    FileNotFoundError for subsequent tests.
    """
    original = os.getcwd()
    yield
    os.chdir(original)
