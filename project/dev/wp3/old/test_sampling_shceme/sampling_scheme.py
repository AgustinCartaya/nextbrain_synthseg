
import numpy as np

# from src.utils.io_utils import load_array_if_path
# from src.utils import data_loader_utils as dl_utils

def load_array_if_path(x):
    return np.load(x) if isinstance(x, str) else np.asarray(x)

def convert_labelmap(labelmap, lut):
    out = np.zeros_like(labelmap)
    for lab, idx in lut.items():
        out[labelmap == lab] = idx
    return out


class SamplingScheme(object):
    def __init__(self, input_labels, sampling_labels, output_lut, **kwargs):

        self.input_labels = load_array_if_path(input_labels) # funcion que lee si no es un np array
        self.input_lut = {lab: it_lab for it_lab, lab in enumerate(self.input_labels)}
        self.n_input_labels = len(input_labels)

        self.sampling_labels = load_array_if_path(sampling_labels)
        self.n_sampling_labels = len(sampling_labels)

        if output_lut is not None:
            self.output_lut = output_lut
            self.output_labels = np.unique(list(output_lut.values()))
            self.n_output_labels = len(self.output_labels)

        else:
            self.output_lut = self.input_lut
            self.output_labels = input_labels
            self.n_output_labels = len(input_labels)

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
        class_weights = np.median( total_binc) / total_binc  # total_binc[np.where(self.input_labels==7)[0][0]] / total_binc # cortical WM  #
        class_weights[np.isinf(class_weights)] = 0

        return class_weights

    @NotImplementedError
    def compute_stats(self, high_level_mus, high_level_sigmas, n_channels=1):
        pass

class ERCHierarchicalSampling(SamplingScheme):
    def __init__(self, input_labels, randomize_less=True, std_noise=8 + np.random.randn(1), output_lut=None):

        self.randomize_less = randomize_less
        self.std_noise = std_noise

        self.dict_sampling_labels = {0: 'MU_GM', 1: 'MU_WM', 2: 'MU_CSF', 3: 'MU_BG',
                                     4: 'MU_DURA', 5: 'MU_SKULL', 6: 'MU_SCALP'}

        print('[GENERATIVE MODEL]: Remember that MU/STD BG,DURA,SKULL,SCALP=0 for all models (T1, T2, random, etc...)')
        super().__init__(input_labels=input_labels,
                         sampling_labels=list(self.dict_sampling_labels.keys()),
                         output_lut=output_lut)


    def compute_stats(self,
                      high_level_mus,
                      high_level_sigmas,
                      n_channels=1):


        ####################
        # MODEL PARAMETERS #
        ####################

        # Increments for eg creating a bit of contrast inside layers etc
        small_inc_mus = 2.0
        small_inc_sigmas = 1.0

        # Increments when sampling eg means/stddevs as linear combinations
        pinch_of_randomness = 1.0

        # Increment used for eg going from cerebral to cerebellar white matter
        big_inc_mus = 7.0
        big_inc_sigmas = 3.0

        # Increment used for eg going from between layers of cerebellar cortex
        # or to sample caudate / etc means from global gray matter
        medium_inc_mus = 5.0
        medium_inc_sigmas = 2.0

        if self.randomize_less:
            small_inc_mus = small_inc_mus / 5
            small_inc_sigmas = small_inc_sigmas / 5
            pinch_of_randomness = pinch_of_randomness / 5
            big_inc_mus = big_inc_mus / 5
            big_inc_sigmas = big_inc_sigmas / 5
            medium_inc_mus = medium_inc_mus / 5
            medium_inc_sigmas = medium_inc_sigmas / 5

        # just a factor to scale them all, so you don't need to change 1 at the time
        factor = 2.0  # this number seems to produce realistic images.
        small_inc_mus = small_inc_mus * factor
        small_inc_sigmas = small_inc_sigmas * factor
        medium_inc_mus = medium_inc_mus * factor
        medium_inc_sigmas = medium_inc_sigmas * factor
        big_inc_mus = big_inc_mus * factor
        big_inc_sigmas = big_inc_sigmas * factor
        pinch_of_randomness = pinch_of_randomness * factor


        ##################
        # HIGH-LEVEL LUT #
        ##################
        MU_GM, MU_WM, MU_CSF, MU_BG, MU_DURA, MU_SKULL, MU_SCALP = high_level_mus

        STD_GM, STD_WM, STD_CSF, STD_BG, STD_DURA, STD_SKULL, STD_SCALP = high_level_sigmas

        MU_BG, MU_DURA, MU_SKULL, MU_SCALP = 0, 0, 0, 0
        STD_BG, STD_DURA, STD_SKULL, STD_SCALP = 0, 0, 0, 0

        #################
        # LOW-LEVEL LUT #
        #################
        low_level_mus = np.zeros(self.n_input_labels)
        low_level_sigmas = np.zeros(self.n_input_labels)

        # Background (the easiest, hehe)
        idx = np.where(self.input_labels == 0)[0]
        low_level_mus[idx] = MU_BG
        low_level_sigmas[idx] = STD_BG

        # Cerebral white matter
        region_labels = [2, 7, 68, 120, 130, 199, 210, 445, 513, 514, 602, 100, 161, 208, 209]
        for lab in region_labels:
            idx = np.where(self.input_labels == lab)[0]
            if any(low_level_mus[idx]):
                raise ValueError('a label has already been assigned')
            low_level_mus[idx] = MU_WM + small_inc_mus * np.random.randn(1)
            low_level_sigmas[idx] = STD_WM + small_inc_sigmas * np.random.randn(1)

        # Cerebral gray matter
        region_labels = [3, 103, 106, 108, 111, 117, 127, 128, 129, 146, 158, 99, 113, 125, 155, 156, 157]
        for lab in region_labels:
            idx = np.where(self.input_labels == lab)[0]
            if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
            low_level_mus[idx] = MU_GM + small_inc_mus * np.random.randn(1)
            low_level_sigmas[idx] = STD_GM + small_inc_sigmas * np.random.randn(1)

        # Lateral and inferio lateral ventricle
        # Also 291 = choroid. For the time being, we merge with ventricles
        region_labels = [50, 383, 620, 688, 328, 291]
        for lab in region_labels:
            idx = np.where(self.input_labels == lab)[0]
            if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
            low_level_mus[idx] = MU_CSF
            low_level_sigmas[idx] = STD_CSF

        # WM cerebellum
        idx = np.where(self.input_labels == 611)[0]
        if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
        low_level_mus[idx] = MU_WM + big_inc_mus * np.random.randn(1)
        low_level_sigmas[idx] = STD_WM + big_inc_sigmas * np.random.randn(1)

        # GM cerebellum
        region_labels = [595, 597]
        mu_gmc = MU_GM + big_inc_mus * np.random.randn(1)
        for lab in region_labels:
            idx = np.where(self.input_labels == lab)[0]
            if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
            low_level_mus[idx] = mu_gmc + medium_inc_mus * np.random.randn(1)
            low_level_sigmas[idx] = STD_GM + medium_inc_sigmas * np.random.randn(1)

        # Caudate and accumbens - GM nuclei
        region_labels = [48, 118, 393, 101]
        mu_cau = MU_GM + medium_inc_mus * np.random.randn(1)
        std_cau = STD_GM + medium_inc_sigmas * np.random.randn(1)
        for lab in region_labels:
            idx = np.where(self.input_labels == lab)[0]
            if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
            low_level_mus[idx] = mu_cau + small_inc_mus * np.random.randn(1)
            low_level_sigmas[idx] = std_cau + small_inc_sigmas * np.random.randn(1)

        # Caudate and accumbens - Other nuclei
        # stria terminalis (184) and bnst (187) should be grouped with pericaudate
        # (49), the are closer to WM
        # We also add  blood vessels of the forebrain (whicn Nellie started
        # labeling but then we said to stop).
        region_labels = [49, 184, 187, 69]
        a = 0.5 if self.randomize_less else 0.2 + 0.6 * np.random.rand(1)
        mu_cau2 = a * MU_GM + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)
        a = 0.5 if self.randomize_less else 0.2 + 0.6 * np.random.rand(1)
        std_cau2 = a * STD_GM + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)
        for lab in region_labels:
            idx = np.where(self.input_labels == lab)[0]
            if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
            low_level_mus[idx] = mu_cau2 + small_inc_mus * np.random.randn(1)
            low_level_sigmas[idx] = std_cau2 + small_inc_sigmas * np.random.randn(1)

        # Putamen
        # Posteroventral putamen (349) modeled with putamen (identical)
        region_labels = [79, 349]
        mu_put = MU_GM + medium_inc_mus * np.random.randn(1)
        std_put = STD_GM + medium_inc_sigmas * np.random.randn(1)
        for lab in region_labels:
            idx = np.where(self.input_labels == lab)
            if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
            low_level_mus[idx] = mu_put
            low_level_sigmas[idx] = std_put

        # Claustrum and temporal claustrum (the latter is more WM-ish)
        idx = np.where(self.input_labels == 102)[0]
        mu_claus = MU_GM + medium_inc_mus * np.random.randn(1)
        std_claus = STD_GM + medium_inc_sigmas * np.random.randn(1)
        if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
        low_level_mus[idx] = mu_claus
        low_level_sigmas[idx] = std_claus

        idx = np.where(self.input_labels == 174)[0]
        a = 0.7 if self.randomize_less else 0.4 + 0.6 * np.random.rand(1)
        if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
        low_level_mus[idx] = a * mu_claus + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)
        a = 0.7 if self.randomize_less else 0.4 + 0.6 * np.random.rand(1)
        low_level_sigmas[idx] = a * std_claus + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)

        # Pallidum: internal lamina (210) goes to WM
        a = 0.75 if self.randomize_less else 0.5 + 0.5 * np.random.rand(1)
        mu_pal_ext = a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)
        a = 0.75 if self.randomize_less else 0.5 + 0.5 * np.random.rand(1)
        std_pal_ext = a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)
        idx = np.where(self.input_labels == 119)[0]
        if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
        low_level_mus[idx] = mu_pal_ext + small_inc_mus * np.random.randn(1)
        low_level_sigmas[idx] = std_pal_ext + small_inc_sigmas * np.random.randn(1)

        a = 0.1 if self.randomize_less else -0.15 + 0.3 * np.random.rand(1)
        mu_pal_int = mu_pal_ext + a * (MU_WM - MU_GM) + pinch_of_randomness * np.random.randn(1)
        std_pal_int = std_pal_ext + pinch_of_randomness * np.random.randn(1)
        idx = np.where(self.input_labels == 206)[0]
        if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
        low_level_mus[idx] = mu_pal_int + small_inc_mus * np.random.randn(1)
        low_level_sigmas[idx] = std_pal_int + small_inc_sigmas * np.random.randn(1)

        # ----------- #
        # Hippocampus #
        # ----------- #

        # (1) WM-ish clusters
        # (1.1) lacunosum moleculare
        region_labels = [343, 368, 372, 408, 418, 562, 566, 571, 339]
        a = 0.8 if self.randomize_less else 0.6 + 0.4 * np.random.rand(1)
        mu = a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)
        a = 0.8 if self.randomize_less else 0.6 + 0.4 * np.random.rand(1)
        std = a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)
        for lab in region_labels:
            idx = np.where(self.input_labels == lab)[0]
            if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
            low_level_mus[idx] = mu
            low_level_sigmas[idx] = std

        # (1.2) alveus
        a = 0.8 if self.randomize_less else 0.6 + 0.4 * np.random.rand(1)
        mu = a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)
        a = 0.8 if self.randomize_less else 0.6 + 0.4 * np.random.rand(1)
        std = a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)
        idx = np.where(self.input_labels == 354)[0]
        if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
        low_level_mus[idx] = mu
        low_level_sigmas[idx] = std

        # (1.3) fimbria
        a = 0.8 if self.randomize_less else 0.6 + 0.4 * np.random.rand(1)
        mu = a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)
        a = 0.8 if self.randomize_less else 0.6 + 0.4 * np.random.rand(1)
        std = a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)
        idx = np.where(self.input_labels == 461)[0]
        if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
        low_level_mus[idx] = mu
        low_level_sigmas[idx] = std

        # (2) Shades of GM
        mu_hippo_gm = MU_GM + medium_inc_mus * np.random.randn(1)
        std_hippo_gm = STD_GM + medium_inc_sigmas * np.random.randn(1)
        region_list = {}
        region_list[0] = [344, 369, 409, 567, 340, 563, 419, 373, 572]  # radiatum
        region_list[1] = [345, 370, 374, 410, 420, 564, 573, 341, 568]  # pyramidale
        region_list[2] = [371, 375, 411, 421, 565, 569, 574, 342, 346]  # oriens
        region_list[3] = [326, 347, 576, 348, 327]  # subiculum + prosubiculum
        region_list[4] = [358]  # presubiculum
        region_list[5] = [558, 404, 364]  # mol layer DG
        region_list[6] = [405, 559, 365]  # granula layer DG
        region_list[7] = [561, 407, 367]  # polyform layer DG
        region_list[8] = [575, 422, 432]  # pyramidal cells CA4
        region_list[9] = [320]  # HATA
        for k, v in region_list.items():
            mu = mu_hippo_gm + small_inc_mus * np.random.randn(1)
            std = std_hippo_gm + small_inc_sigmas * np.random.randn(1)
            for lab in v:
                idx = np.where(self.input_labels == lab)[0]
                if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
                low_level_mus[idx] = mu
                low_level_sigmas[idx] = std

        # -------- #
        # Thalamus #
        # -------- #

        # Juri created a scale 1-5 where 1 is almost WM and 5 is almost GM
        # We first sample intensities for those 5
        mu_th = []
        a = 0.85 if self.randomize_less else 0.7 + 0.3 * np.random.rand(1)
        mu_th += [a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.675 if self.randomize_less else 0.525 + 0.3 * np.random.rand(1)
        mu_th += [a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.5 if self.randomize_less else 0.35 + 0.3 * np.random.rand(1)
        mu_th += [a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.325 if self.randomize_less else 0.175 + 0.3 * np.random.rand(1)
        mu_th += [a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.15 if self.randomize_less else 0.0 + 0.3 * np.random.rand(1)
        mu_th += [a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)]

        std_th = []
        a = 0.85 if self.randomize_less else 0.7 + 0.3 * np.random.rand(1)
        std_th += [a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.675 if self.randomize_less else 0.525 + 0.3 * np.random.rand(1)
        std_th += [a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.5 if self.randomize_less else 0.35 + 0.3 * np.random.rand(1)
        std_th += [a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.325 if self.randomize_less else 0.175 + 0.3 * np.random.rand(1)
        std_th += [a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.15 if self.randomize_less else 0.0 + 0.3 * np.random.rand(1)
        std_th += [a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)]

        region_list = {}  # we fill this in following Juri's spreadsheet (Eugenio and Nellie added 459, 484 = LGN to 2)
        region_list[0] = [276, 444, 430, 809, 815]
        region_list[1] = [201, 314, 381, 382, 394, 458, 283, 493, 519, 812, 477, 254, 191, 312, 285, 425, 284, 396, 479,
                          350, 222,
                          397, 459, 484]
        region_list[2] = [218, 492, 322, 442, 190, 818, 246, 200, 517, 478, 505, 226, 303, 398, 426, 424, 317]
        region_list[3] = [400, 274, 395, 423, 811, 441, 823, 504, 511, 512, 510, 378, 221, 379, 313, 282, 225, 808, 224,
                          399, 286,
                          223, 454, 380, 296, 814, 816]
        region_list[4] = [813, 219, 220, 252, 253, 443, 508]

        for k, v in region_list.items():
            mu = mu_th[k]
            std = std_th[k]
            for lab in v:
                idx = np.where(self.input_labels == lab)[0]
                if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
                low_level_mus[idx] = mu + small_inc_mus * np.random.randn(1)
                low_level_sigmas[idx] = std + small_inc_sigmas * np.random.randn(1)

        # -------------------- #
        # Brainstem (midbrain) #
        # -------------------- #

        # Similar to thalamus, but with 4 levels
        mu_bs = []
        a = 0.85 if self.randomize_less else 0.7 + 0.3 * np.random.rand(1)
        mu_bs += [a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.675 if self.randomize_less else 0.525 + 0.3 * np.random.rand(1)
        mu_bs += [a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.5 if self.randomize_less else 0.35 + 0.3 * np.random.rand(1)
        mu_bs += [a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.325 if self.randomize_less else 0.175 + 0.3 * np.random.rand(1)
        mu_bs += [a * MU_WM + (1 - a) * MU_GM + pinch_of_randomness * np.random.randn(1)]

        std_bs = []
        a = 0.85 if self.randomize_less else 0.7 + 0.3 * np.random.rand(1)
        std_bs += [a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.675 if self.randomize_less else 0.525 + 0.3 * np.random.rand(1)
        std_bs += [a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.5 if self.randomize_less else 0.35 + 0.3 * np.random.rand(1)
        std_bs += [a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)]
        a = 0.325 if self.randomize_less else 0.175 + 0.3 * np.random.rand(1)
        std_bs += [a * STD_WM + (1 - a) * STD_GM + pinch_of_randomness * np.random.randn(1)]

        region_list = {}  # we fill this in following Juri's spreadsheet
        region_list[0] = [385, 472, 525]
        region_list[1] = [451, 526]
        region_list[2] = [384, 401, 431, 467, 468, 469, 470, 496, 498, 403, 352, 449, 507, 386]
        region_list[3] = [310, 465]

        for k, v in region_list.items():
            mu = mu_bs[k]
            std = std_bs[k]
            for lab in v:
                idx = np.where(self.input_labels == lab)[0]
                if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
                low_level_mus[idx] = mu + small_inc_mus * np.random.randn(1)
                low_level_sigmas[idx] = std + small_inc_sigmas * np.random.randn(1)

        # ------------ #
        # Hypothalamus #
        # ------------ #

        # Nelie did something similar to what Juri did for the thalamus.
        #  1 is the darkest/grey matter and 5 is the brightest/white matter.
        # However, rather than being evenly distributed between WM and GM, lists 2-4
        # could be shifted slightly more towards GM, i.e. no overlap between 4 and 5, if that makes sense.
        mu_hy = []
        a = 1.0 if self.randomize_less else 0.85 + 0.3 * np.random.rand(1)
        mu_hy += [a * MU_GM + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.8 if self.randomize_less else 0.65 + 0.3 * np.random.rand(1)
        mu_hy += [a * MU_GM + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.6 if self.randomize_less else 0.45 + 0.3 * np.random.rand(1)
        mu_hy += [a * MU_GM + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.4 if self.randomize_less else 0.25 + 0.3 * np.random.rand(1)
        mu_hy += [a * MU_GM + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.2 if self.randomize_less else 0.0 + 0.4 * np.random.rand(1)
        mu_hy += [a * MU_GM + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)]

        std_hy = []
        a = 1.0 if self.randomize_less else 0.85 + 0.3 * np.random.rand(1)
        std_hy += [a * STD_GM + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.8 if self.randomize_less else 0.65 + 0.3 * np.random.rand(1)
        std_hy += [a * STD_GM + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.6 if self.randomize_less else 0.45 + 0.3 * np.random.rand(1)
        std_hy += [a * STD_GM + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.4 if self.randomize_less else 0.25 + 0.3 * np.random.rand(1)
        std_hy += [a * STD_GM + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.2 if self.randomize_less else 0.0 + 0.4 * np.random.rand(1)
        std_hy += [a * STD_GM + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)]

        region_list = {}
        region_list[0] = [256, 275, 287, 304]
        region_list[1] = [147, 150, 164, 179, 192, 193, 194, 195, 197, 207, 227, 228, 229, 243, 244, 245, 255, 268, 297,
                          308]
        region_list[2] = [149, 160, 181, 182, 196, 198, 230, 231, 232, 269, 305, 306, 307, 333, 351]
        region_list[3] = [316, 321]
        region_list[4] = [234, 289, 298, 309, 315, 325, 433]

        for k, v in region_list.items():
            mu = mu_hy[k]
            std = std_hy[k]
            for lab in v:
                idx = np.where(self.input_labels == lab)[0]
                if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
                low_level_mus[idx] = mu + small_inc_mus * np.random.randn(1)
                low_level_sigmas[idx] = std + small_inc_sigmas * np.random.randn(1)

        # -------- #
        # Amygdala #
        # -------- #

        # Nellie: There's 3 major groups but we also want to make sure the following\
        # clusters have the same Gaussians:
        # cortical nucleus: 238,249,292,300
        # medial nucleus: 279,319
        # central:277,278,376

        mu_am = []
        a = 1.0 if self.randomize_less else 0.875 + 0.3 * np.random.rand(1)
        mu_am += [a * MU_GM + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.925 if self.randomize_less else 0.80 + 0.2 * np.random.rand(1)
        mu_am += [a * MU_GM + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.85 if self.randomize_less else 0.725 + 0.2 * np.random.rand(1)
        mu_am += [a * MU_GM + (1 - a) * MU_WM + pinch_of_randomness * np.random.randn(1)]

        std_am = []
        a = 1.0 if self.randomize_less else 0.875 + 0.3 * np.random.rand(1)
        std_am += [a * STD_GM + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.925 if self.randomize_less else 0.80 + 0.2 * np.random.rand(1)
        std_am += [a * STD_GM + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)]
        a = 0.85 if self.randomize_less else 0.725 + 0.2 * np.random.rand(1)
        std_am += [a * STD_GM + (1 - a) * STD_WM + pinch_of_randomness * np.random.randn(1)]

        region_list = {}
        region_list[0] = [238, 249, 277, 278, 279, 292, 300, 301, 319, 376, 377, 217, 240, 242]
        region_list[1] = [214, 215, 216]
        region_list[2] = [295]

        for k, v in region_list.items():
            mu = mu_am[k]
            std = std_am[k]
            for lab in v:
                idx = np.where(self.input_labels == lab)[0]
                if any(low_level_mus[idx]): raise ValueError('a label has already been assigned')
                low_level_mus[idx] = mu + small_inc_mus * np.random.randn(1)
                low_level_sigmas[idx] = std + small_inc_sigmas * np.random.randn(1)

        region_labels = [249, 292, 300]
        idx = np.where(self.input_labels == 238)[0]
        for lab in region_labels:
            idx2 = np.where(self.input_labels == lab)[0]
            low_level_mus[idx2] = low_level_mus[idx]
            low_level_sigmas[idx2] = low_level_sigmas[idx]

        region_labels = [319]
        idx = np.where(self.input_labels == 279)[0]
        for lab in region_labels:
            idx2 = np.where(self.input_labels == lab)[0]
            low_level_mus[idx2] = low_level_mus[idx]
            low_level_sigmas[idx2] = low_level_sigmas[idx]

        region_labels = [278, 376]
        idx = np.where(self.input_labels == 277)[0]
        for lab in region_labels:
            idx2 = np.where(self.input_labels == lab)[0]
            low_level_mus[idx2] = low_level_mus[idx]
            low_level_sigmas[idx2] = low_level_sigmas[idx]

        # Ugly temporary fixes to remove when label maps have been cleaned/fixed.
        # (i) Some voxel of red nucleus (385) were labeled as 1
        i = np.where(self.input_labels == 1)[0]
        j = np.where(self.input_labels == 385)[0]
        low_level_mus[i] = low_level_mus[j]
        low_level_sigmas[i] = low_level_sigmas[j]

        # (ii) 500: it's a generic hippocampus label that should not be there. Nellie will double check
        # for now, wel fill in with 575 = pyramidal layers of CA4
        i = np.where(self.input_labels == 500)[0]
        j = np.where(self.input_labels == 575)[0]
        low_level_mus[i] = low_level_mus[j]
        low_level_sigmas[i] = low_level_sigmas[j]

        low_level_sigmas = np.sqrt(low_level_sigmas ** 2 + self.std_noise ** 2)

        return low_level_mus, low_level_sigmas