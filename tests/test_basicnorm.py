from audio_preprocess import BasicNorm
from audio_preprocess.basicnorm import audio_to_mono
import librosa
import os
import torch
import soundfile as sf
import numpy as np
import pytest
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLE_WAV = os.path.join(BASE_DIR,"data/LA_T_5260333.wav")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def test_load_audio():
    bnp = BasicNorm()
    # check unsupported format
    with pytest.raises(Exception) as excinfo:
        _, _ = bnp.load_audio(os.path.join(BASE_DIR,"data/LA_T_5260333.mp4"))
    assert "Unsupported file format" in str(excinfo.value)
    
    # check load supported format
    audio, sr = bnp.load_audio(SAMPLE_WAV)
    # check if load audio without full 0
    assert audio.shape[0] > 0
    assert not np.allclose(audio, 0)
    

def test_basicProcess():
    bnp = BasicNorm()
    # load 2 channel audio
    audio, sr = bnp.load_audio(os.path.join(BASE_DIR,"data/LA_T_5260333_2.wav"))
    # check original audio is stereo
    assert len(audio.shape) == 2
    out_signal = bnp.process(audio)
    # check if audio is mono and not all 0
    assert len(out_signal.shape) == 1
    assert out_signal.shape[0] > 0
    assert not np.allclose(out_signal, 0)
    


def test_audio_to_mono():
    # Generate a stereo audio signal
    stereo_signal = np.random.rand(2, 1000)
    # Convert to mono
    mono_signal, sr = audio_to_mono(stereo_signal)
    # Check that the resulting signal is mono
    assert mono_signal.shape == (1000,)
    # Check that the resulting signal is not all zeros
    assert not np.allclose(mono_signal, 0)