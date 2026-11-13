"""Configuration: YAML files with inheritance, command-line overrides and path resolution."""
import argparse
import copy
from pathlib import Path

import yaml

#: the project folder (the one that contains configs/ and the numbered scripts); relative paths of the `paths` section start here
ROOT = Path(__file__).resolve().parents[1]


def _merge(base, new):
    """Recursively update dict `base` with dict `new` (in place) and return it."""
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _read(path):
    """Read one YAML file, first loading the file named by its `_base_` key (if any)."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is None:
        return cfg
    return _merge(_read(path.parent / base), cfg)


def apply_overrides(cfg, overrides):
    """Apply `a.b.c=value` strings to `cfg`. Values are parsed as YAML (numbers, lists, bools)."""
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def resolve_paths(cfg):
    """Make every entry of cfg['paths'] absolute (relative ones are taken from ROOT)."""
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else (ROOT / p).resolve())
    return cfg


def load_config(path, overrides=None):
    """Load a config file -> nested dict with absolute paths.

    path:      YAML file (may inherit from another one through `_base_`)
    overrides: list of "section.key=value" strings (the `--set` option of the scripts)
    """
    return resolve_paths(apply_overrides(_read(path), overrides))


def add_config_args(parser):
    """Add the two options shared by all scripts: --config and --set."""
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"),
                        help="YAML configuration file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="overrides, e.g. --set train.batch_size=4 data.max_voxel_mm=0.8")
    return parser


def config_from_args(args):
    """Config dict from the parsed --config / --set options."""
    return load_config(args.config, args.set)


def checkpoint_config(ckpt_cfg, cfg):
    """Config to use with a trained model.

    The sections that define the network and its input (labels, model, patch,
    data.target_spacing) come from the checkpoint, so they always match the
    weights; everything else (paths, inference, evaluation) from `cfg`.
    """
    out = copy.deepcopy(cfg)
    for key in ("labels", "model", "patch"):
        out[key] = copy.deepcopy(ckpt_cfg[key])
    out["data"]["target_spacing"] = ckpt_cfg["data"]["target_spacing"]
    return out


def parser(description):
    """ArgumentParser with the module docstring as help text and --config / --set."""
    return add_config_args(argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter))
