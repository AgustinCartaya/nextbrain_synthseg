

[
    {'name': 'a', 
    'children': [
        {'name': 'a-a', 
        'children': [
            {'name': 'a-a-a', 'children': []},
        ]
        }
    ],
    },
    
]



[
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
]