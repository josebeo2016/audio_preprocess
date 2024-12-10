import pytest
import numpy as np
from audio_preprocess.denoise_semamba import DeNoise_SEMamba
import soundfile as sf
import os
import librosa

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLE_WAV = os.path.join(BASE_DIR,"data/LA_T_5260333.wav")

@pytest.mark.filterwarnings
def test_process():
    # Create a sample input signal
    input_signal, sr = librosa.load(SAMPLE_WAV, sr=16000)

    # Create an instance of the DeNoise class
    de_noiser = DeNoise_SEMamba()

    # Process the input signal
    output_signal = de_noiser.process(input_signal)
    # manually check
    sf.write(os.path.join(BASE_DIR,"data/LA_T_5260333_denoised_semamba.wav"), output_signal, sr) 

    # Check if the output signal has the same length as the input signal
    assert abs(len(output_signal) - len(input_signal)) < 1000

    # Check if the output signal is a numpy array
    assert isinstance(output_signal, np.ndarray)

    # Check if the output signal is not empty
    assert output_signal.size > 0

    # Add more assertions as needed to validate the output signal


if __name__ == '__main__':
    pytest.main()
