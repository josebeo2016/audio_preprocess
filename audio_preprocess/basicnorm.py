import librosa
import numpy as np
import soundfile as sf

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
    def _supported_format(self) -> list:
        return list(sf.available_formats().keys())
    def _supported_subtypes(self, format: str) -> list:
        return list(sf.available_subtypes(format).keys())
    
    def load_audio(self, input_path: str, sr=16000) -> np.ndarray:
        """
        Load audio from file path. Default sample rate is 16000.
        Any pysoundfile supported format is supported. 
        Example: wav, flac, mp3, ogg, etc.
        """
        file_format = input_path.split('.')[-1]
        if file_format.upper() not in self._supported_format():
            raise ValueError(f"Unsupported file format: {file_format}")
        audio, sr = librosa.load(input_path, sr=sr, mono=False)
        return audio, sr
        
    
    def process(self, input_signal: np.ndarray, sr=16000) -> np.ndarray:
        # convert to mono
        if (len(input_signal.shape) !=1):
            output_signal, sr = audio_to_mono(input_signal, sr)
        else:
            output_signal = input_signal
        # normalize
        # [TODO]
        return output_signal