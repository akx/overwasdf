import os

import librosa
import numpy as np


def load_vectors(audio_filenames, sample_rate):
    audio_vectors = {}
    for filename in audio_filenames:
        print(filename)
        vec = librosa.load(filename, sr=sample_rate, mono=True)[0]
        # Normalize:
        v_max = np.max(vec)
        v_min = np.min(vec)
        vec = (vec - v_min) / (v_max / v_min)
        audio_vectors[os.path.basename(filename)] = vec
    return audio_vectors
