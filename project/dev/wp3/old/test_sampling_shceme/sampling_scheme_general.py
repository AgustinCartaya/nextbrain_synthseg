"""
Generic hierarchical intensity sampling scheme.

Unlike `sampling_scheme.ERCHierarchicalSampling`, nothing here is tied to a
specific labelling protocol. The label -> intensity model is described by a
config (a python dict or a JSON file) made of:

  tissues        : names of the high-level classes, in the same order as the
                   `high_level_mus` / `high_level_sigmas` passed to compute_stats.
  zero_tissues   : tissues whose mean/std are forced to 0 (e.g. background).
  groups         : ordered list of groups. Each group samples one (mu, std) pair
                   and optionally assigns it (+ per-label jitter) to some labels.
                   A group's `base` can be a tissue or a previously defined group,
                   which gives the hierarchy (e.g. cerebellum GM -> its layers).
  same_as        : {label: reference_label} aliases, copied at the end.
  unassigned     : what to do with input labels not covered by any group:
                   'random' (SynthSeg-like independent sampling), 'zero' or 'error'.

Group spec (all keys optional except `name`):
  {
    'name': 'thalamus',
    'labels': [10, 49],
    # either a single base ...
    'base': 'GM',                     # tissue name or previous group name
    # ... or a linear mixture of two bases:  w * A + (1 - w) * B
    'mix': ['WM', 'GM'],
    'weight': [0.35, 0.65],           # float, or [lo, hi] sampled uniformly
                                      # (midpoint when randomize_less=True)
    'group_inc': 'medium',            # jitter of the group w.r.t. its base
    'label_inc': 'small',             # jitter of each label w.r.t. the group
  }
Increment levels: 'none', 'pinch', 'small', 'medium', 'big'.
"""

import json
import warnings

import numpy as np


def load_array_if_path(x):
    return np.load(x) if isinstance(x, str) else np.asarray(x)


def load_config_if_path(x):
    if isinstance(x, str):
        with open(x) as f:
            return json.load(f)
    return x


def convert_labelmap(labelmap, lut, default=0):
    """Map raw label values to indices using `lut` ({raw_label: index}). Unknown labels -> `default`."""
    labelmap = np.asarray(labelmap).astype(int)
    keys = np.array(sorted(lut.keys()), dtype=int)
    vals = np.array([lut[k] for k in keys], dtype=int)
    pos = np.clip(np.searchsorted(keys, labelmap), 0, len(keys) - 1)
    found = keys[pos] == labelmap
    return np.where(found, vals[pos], default)


# ------------------------------------------------------------------------------------------------ #
# Presets                                                                                          #
# ------------------------------------------------------------------------------------------------ #

# FreeSurfer / SynthSeg / SuperSynth label protocol (left and right hemispheres share a group).
FREESURFER_CONFIG = {
    'tissues': ['GM', 'WM', 'CSF', 'BG'],
    'zero_tissues': ['BG'],
    'unassigned': 'random',
    'groups': [
        {'name': 'background', 'labels': [0], 'base': 'BG'},

        # Cerebrum
        {'name': 'cerebral_wm', 'labels': [2, 41], 'base': 'WM', 'label_inc': 'small'},
        {'name': 'cerebral_gm', 'labels': [3, 42], 'base': 'GM', 'label_inc': 'small'},
        {'name': 'wm_hypointensities', 'labels': [77, 78, 79],
         'mix': ['WM', 'GM'], 'weight': [0.3, 0.7], 'label_inc': 'small'},

        # CSF spaces (choroid plexus merged with ventricles, as in the ERC scheme)
        {'name': 'csf', 'labels': [4, 5, 14, 15, 24, 31, 43, 44, 63, 72], 'base': 'CSF'},

        # Cerebellum
        {'name': 'cerebellar_wm', 'labels': [7, 46], 'base': 'WM', 'group_inc': 'big', 'label_inc': 'small'},
        {'name': 'cerebellar_gm', 'labels': [8, 47], 'base': 'GM', 'group_inc': 'big', 'label_inc': 'small'},

        # Deep GM
        {'name': 'thalamus', 'labels': [10, 49],
         'mix': ['WM', 'GM'], 'weight': [0.35, 0.65], 'label_inc': 'small'},
        {'name': 'caudate_accumbens', 'labels': [11, 50, 26, 58],
         'base': 'GM', 'group_inc': 'medium', 'label_inc': 'small'},
        {'name': 'putamen', 'labels': [12, 51], 'base': 'GM', 'group_inc': 'medium', 'label_inc': 'small'},
        {'name': 'pallidum', 'labels': [13, 52],
         'mix': ['WM', 'GM'], 'weight': [0.5, 1.0], 'label_inc': 'small'},
        {'name': 'ventral_dc', 'labels': [28, 60],
         'mix': ['WM', 'GM'], 'weight': [0.5, 0.9], 'label_inc': 'small'},

        # Medial temporal lobe
        {'name': 'hippocampus', 'labels': [17, 53], 'base': 'GM', 'group_inc': 'medium', 'label_inc': 'small'},
        {'name': 'amygdala', 'labels': [18, 54], 'base': 'GM', 'group_inc': 'medium', 'label_inc': 'small'},

        # Brainstem and others
        {'name': 'brainstem', 'labels': [16],
         'mix': ['WM', 'GM'], 'weight': [0.6, 0.9], 'label_inc': 'small'},
        {'name': 'optic_chiasm', 'labels': [85], 'base': 'WM', 'group_inc': 'small'},
        {'name': 'vessel', 'labels': [30, 62], 'mix': ['GM', 'CSF'], 'weight': [0.2, 0.8]},
    ],
    'same_as': {},
}

PRESETS = {
    'freesurfer': FREESURFER_CONFIG,
    'synthseg': FREESURFER_CONFIG,
}


# ------------------------------------------------------------------------------------------------ #
# Sampling schemes                                                                                 #
# ------------------------------------------------------------------------------------------------ #

class SamplingScheme(object):
    def __init__(self, input_labels, sampling_labels, output_lut=None, **kwargs):

        self.input_labels = load_array_if_path(input_labels).astype(int)
        self.input_lut = {int(lab): it_lab for it_lab, lab in enumerate(self.input_labels)}
        self.n_input_labels = len(self.input_labels)

        self.sampling_labels = list(sampling_labels)
        self.n_sampling_labels = len(self.sampling_labels)

        if output_lut is not None:
            self.output_lut = output_lut
            self.output_labels = np.unique(list(output_lut.values()))
            self.n_output_labels = len(self.output_labels)
        else:
            self.output_lut = self.input_lut
            self.output_labels = self.input_labels
            self.n_output_labels = self.n_input_labels

    def convert_labelmap(self, labelmap, lut=None):
        if lut is None:
            lut = self.input_lut
        return convert_labelmap(labelmap, lut).astype('int')

    def get_class_weights(self, subject_dict):
        binc = []
        for subject in subject_dict.values():
            clabel = self.convert_labelmap(np.asarray(subject.labelmap.dataobj))
            binc.append(np.bincount(clabel.reshape(-1, ), minlength=self.n_input_labels))

        total_binc = sum(binc) / len(subject_dict)
        with np.errstate(divide='ignore'):
            class_weights = np.median(total_binc) / total_binc
        class_weights[np.isinf(class_weights)] = 0

        return class_weights

    def compute_stats(self, high_level_mus, high_level_sigmas, n_channels=1):
        raise NotImplementedError


class GenericHierarchicalSampling(SamplingScheme):
    """Hierarchical sampling driven by a config (see module docstring) instead of hardcoded labels.

    Example:
        scheme = GenericHierarchicalSampling(labels, config='freesurfer')
        scheme = GenericHierarchicalSampling(labels, config='my_protocol.json')
        scheme = GenericHierarchicalSampling.from_tissue_map(labels, {2: 'WM', 3: 'GM', 4: 'CSF'})
        mus, sigmas = scheme.compute_stats(high_mus, high_sigmas)
        image = mus[idx] + sigmas[idx] * randn, with idx = scheme.convert_labelmap(seg)
    """

    INC_LEVELS = ('none', 'pinch', 'small', 'medium', 'big')

    def __init__(self, input_labels, config='freesurfer', randomize_less=True, std_noise=None,
                 output_lut=None, inc_factor=2.0, seed=None, verbose=True):

        if isinstance(config, str) and config.lower() in PRESETS:
            config = PRESETS[config.lower()]
        self.config = load_config_if_path(config)

        self.tissues = list(self.config.get('tissues', ['GM', 'WM', 'CSF', 'BG']))
        self.zero_tissues = set(self.config.get('zero_tissues', []))
        self.groups = list(self.config.get('groups', []))
        self.same_as = {int(k): int(v) for k, v in self.config.get('same_as', {}).items()}
        self.unassigned = self.config.get('unassigned', 'random')
        if self.unassigned not in ('random', 'zero', 'error'):
            raise ValueError(f"unassigned must be 'random', 'zero' or 'error', got {self.unassigned}")

        self.randomize_less = randomize_less
        self.rng = np.random.default_rng(seed)
        self.std_noise = 8 + self.rng.standard_normal() if std_noise is None else std_noise
        self.verbose = verbose
        self.inc_mus, self.inc_sigmas = self._build_increments(randomize_less, inc_factor)

        self.dict_sampling_labels = dict(enumerate(self.tissues))
        super().__init__(input_labels=input_labels,
                         sampling_labels=list(self.dict_sampling_labels.keys()),
                         output_lut=output_lut)

        self._validate_config()

    @classmethod
    def from_tissue_map(cls, input_labels, tissue_map, tissues=('GM', 'WM', 'CSF', 'BG'),
                        zero_tissues=('BG',), label_inc='small', unassigned='random', **kwargs):
        """Simplest usage: {label: tissue_name}. Each label gets its tissue's Gaussian + `label_inc` jitter."""
        by_tissue = {}
        for lab, tissue in tissue_map.items():
            by_tissue.setdefault(tissue, []).append(int(lab))

        config = {
            'tissues': list(tissues),
            'zero_tissues': list(zero_tissues),
            'unassigned': unassigned,
            'groups': [{'name': f'{t}_labels', 'labels': labs, 'base': t,
                        'label_inc': 'none' if t in zero_tissues else label_inc}
                       for t, labs in by_tissue.items()],
        }
        return cls(input_labels, config=config, **kwargs)

    # -------------------------------------------------------------------------------------------- #
    # Setup                                                                                        #
    # -------------------------------------------------------------------------------------------- #

    @staticmethod
    def _build_increments(randomize_less, factor):
        # Same model parameters as ERCHierarchicalSampling
        inc_mus = {'none': 0.0, 'pinch': 1.0, 'small': 2.0, 'medium': 5.0, 'big': 7.0}
        inc_sigmas = {'none': 0.0, 'pinch': 1.0, 'small': 1.0, 'medium': 2.0, 'big': 3.0}

        scale = factor / 5 if randomize_less else factor
        return ({k: v * scale for k, v in inc_mus.items()},
                {k: v * scale for k, v in inc_sigmas.items()})

    def _validate_config(self):
        names = set(self.tissues)
        seen_labels = {}
        for g in self.groups:
            name = g.get('name')
            if name is None:
                raise ValueError(f'Group without name: {g}')
            if name in names:
                raise ValueError(f'Duplicated group/tissue name: {name}')

            if ('base' in g) == ('mix' in g):
                raise ValueError(f"Group '{name}' must define exactly one of 'base' or 'mix'")
            refs = [g['base']] if 'base' in g else list(g['mix'])
            if 'mix' in g and len(refs) != 2:
                raise ValueError(f"Group '{name}': 'mix' must have exactly two entries")
            for r in refs:
                if r not in names:
                    raise ValueError(f"Group '{name}' references '{r}', which is not a tissue "
                                     f"or a previously defined group")

            for key in ('group_inc', 'label_inc'):
                if g.get(key, 'none') not in self.INC_LEVELS:
                    raise ValueError(f"Group '{name}': {key} must be one of {self.INC_LEVELS}")

            for lab in g.get('labels', []):
                if lab in seen_labels:
                    raise ValueError(f"Label {lab} assigned to both '{seen_labels[lab]}' and '{name}'")
                seen_labels[lab] = name

            names.add(name)

        self.label_to_group = seen_labels
        present = set(self.input_labels.tolist())
        self.unassigned_labels = sorted(present - set(seen_labels) - set(self.same_as))

        if self.unassigned_labels:
            msg = (f'[GENERATIVE MODEL]: {len(self.unassigned_labels)} input labels not covered by the config '
                   f'({self.unassigned} policy): {self.unassigned_labels}')
            if self.unassigned == 'error':
                raise ValueError(msg)
            if self.verbose:
                print(msg)

        if self.verbose and self.zero_tissues:
            print(f'[GENERATIVE MODEL]: MU/STD of {sorted(self.zero_tissues)} are forced to 0')

    # -------------------------------------------------------------------------------------------- #
    # Sampling                                                                                     #
    # -------------------------------------------------------------------------------------------- #

    def _randn(self):
        return self.rng.standard_normal()

    def _weight(self, w):
        if np.isscalar(w):
            return float(w)
        lo, hi = w
        return 0.5 * (lo + hi) if self.randomize_less else lo + (hi - lo) * self.rng.random()

    def _sample_group(self, g, table):
        g_inc = g.get('group_inc', 'none')

        if 'base' in g:
            mu_b, std_b = table[g['base']]
            mu = mu_b + self.inc_mus[g_inc] * self._randn()
            std = std_b + self.inc_sigmas[g_inc] * self._randn()
        else:
            (mu_a, std_a), (mu_b, std_b) = table[g['mix'][0]], table[g['mix'][1]]
            w = g.get('weight', [0.0, 1.0])
            pinch = self.inc_mus['pinch']
            a = self._weight(w)
            mu = a * mu_a + (1 - a) * mu_b + pinch * self._randn()
            a = self._weight(w)
            std = a * std_a + (1 - a) * std_b + pinch * self._randn()
            mu += self.inc_mus[g_inc] * self._randn()
            std += self.inc_sigmas[g_inc] * self._randn()

        return mu, std

    def compute_stats(self, high_level_mus, high_level_sigmas, n_channels=1):
        high_level_mus = np.asarray(high_level_mus, dtype=float).reshape(-1)
        high_level_sigmas = np.asarray(high_level_sigmas, dtype=float).reshape(-1)
        if len(high_level_mus) != len(self.tissues) or len(high_level_sigmas) != len(self.tissues):
            raise ValueError(f'Expected {len(self.tissues)} high-level values ({self.tissues}), '
                             f'got {len(high_level_mus)} mus and {len(high_level_sigmas)} sigmas')

        # High-level LUT
        table = {}
        for t, mu, std in zip(self.tissues, high_level_mus, high_level_sigmas):
            table[t] = (0.0, 0.0) if t in self.zero_tissues else (mu, std)

        # Low-level LUT
        low_level_mus = np.zeros(self.n_input_labels)
        low_level_sigmas = np.zeros(self.n_input_labels)
        assigned = np.zeros(self.n_input_labels, dtype=bool)

        for g in self.groups:
            mu_g, std_g = self._sample_group(g, table)
            table[g['name']] = (mu_g, std_g)

            l_inc = g.get('label_inc', 'none')
            for lab in g.get('labels', []):
                if lab not in self.input_lut:
                    continue
                idx = self.input_lut[lab]
                low_level_mus[idx] = mu_g + self.inc_mus[l_inc] * self._randn()
                low_level_sigmas[idx] = std_g + self.inc_sigmas[l_inc] * self._randn()
                assigned[idx] = True

        # Labels not described by the config
        if self.unassigned_labels and self.unassigned == 'random':
            active = [table[t] for t in self.tissues if t not in self.zero_tissues]
            mus_act = np.array([m for m, _ in active]) if active else high_level_mus
            stds_act = np.array([s for _, s in active]) if active else high_level_sigmas
            for lab in self.unassigned_labels:
                idx = self.input_lut[lab]
                low_level_mus[idx] = self.rng.uniform(mus_act.min(), mus_act.max())
                low_level_sigmas[idx] = self.rng.uniform(stds_act.min(), stds_act.max())
                assigned[idx] = True

        # Aliases
        for lab, ref in self.same_as.items():
            if lab in self.input_lut and ref in self.input_lut:
                low_level_mus[self.input_lut[lab]] = low_level_mus[self.input_lut[ref]]
                low_level_sigmas[self.input_lut[lab]] = low_level_sigmas[self.input_lut[ref]]

        low_level_sigmas = np.sqrt(low_level_sigmas ** 2 + self.std_noise ** 2)

        return low_level_mus, low_level_sigmas
