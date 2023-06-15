import pytest
import numpy as np
from audio_preprocess.denoise_audio import DeNoise

@pytest.mark.filterwarnings
def test_process():
    # Create a sample input signal
    input_signal = np.random.rand(10000)

    # Create an instance of the DeNoise class
    de_noiser = DeNoise(input_signal)

    # Process the input signal
    output_signal = de_noiser.process()

    # Check if the output signal has the same length as the input signal
    assert len(output_signal) == len(input_signal)

    # Check if the output signal is a numpy array
    assert isinstance(output_signal, np.ndarray)

    # Check if the output signal is not empty
    assert output_signal.size > 0

    # Add more assertions as needed to validate the output signal


if __name__ == '__main__':
    pytest.main()
