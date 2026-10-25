"""
nxgen: modular synthetic brain MRI generator (WP3, SynthSeg + NextBrain).

Offline:  nxgen.offline.template.build_template   label map -> HR template
Online:   nxgen.pipeline.Generator                HR templates -> images
Training: nxgen.training.SyntheticStream          endless stream of samples

See README.md and docs/ for the documentation.
"""

from .core.config import load_config
from .core.recorder import Recorder
from .pipeline import Generator, Sample

__all__ = ['Generator', 'Sample', 'Recorder', 'load_config']
