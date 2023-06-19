import librosa
import numpy as np


def audio_to_mono(data: np.ndarray, sr: int = 16000) -> np.ndarray:
    y_mono = librosa.to_mono(data)
    return y_mono, sr

class BasicNorm():
    def __init__(self, config=None) -> None:
        """
        Require config:
        None
        """
        self.config = config
        
    def process(self, input_signal: np.ndarray, sr=16000) -> np.ndarray:
        # convert to mono
        output_signal, sr = audio_to_mono(input_signal, sr)
        
        #
        
        return output_signal