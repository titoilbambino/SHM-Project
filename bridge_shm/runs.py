"""Experiment configuration and run artifacts."""

import json
from pathlib import Path
from tempfile import mkdtemp

import yaml


def read_config(path):
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def create_run_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix="run-", dir=parent))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
