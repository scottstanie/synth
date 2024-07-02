from __future__ import annotations

import importlib.metadata

import synth as m


def test_version():
    assert importlib.metadata.version("synth") == m.__version__
