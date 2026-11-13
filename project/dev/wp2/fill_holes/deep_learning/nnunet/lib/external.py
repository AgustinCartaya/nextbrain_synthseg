"""
Import the two existing scripts as libraries, from the paths given in the config.

  corrupt_nifti.py  the corruption generator (used as is, never reimplemented)
  merge_labels.py   the ontology label merger

They live outside this project, so they are loaded by file path. Each file is
imported once per process and then reused.
"""
import importlib.util
import sys

_loaded = {}


def _import_file(name, path):
    """Import the Python file `path` as module `name` (cached)."""
    if path not in _loaded:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _loaded[path] = module
    return _loaded[path]


def corrupt_module(cfg):
    """The corrupt_nifti module (`corrupt_array`, `brain_mask`, `bbox`, ...)."""
    return _import_file("corrupt_nifti", cfg["paths"]["corrupt_script"])


def merge_module(cfg):
    """The merge_labels module (`load_ontology`, `build_label_map`, `build_lut`, ...)."""
    return _import_file("merge_labels", cfg["paths"]["merge_script"])
