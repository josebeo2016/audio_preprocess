import pytest
import numpy as np
from audio_preprocess.dereverb import DeReverb

@pytest.fixture
def dereverb():
    return DeReverb()

def test_process(dereverb):
    # Test input signal
    inputSignal = np.random.randn(16000)

    # Process the input signal
    outputSignal = dereverb.process(inputSignal)

    # Check if the output signal has the correct shape
    assert outputSignal.shape == inputSignal.shape

    # Check if the output signal is not all zeros
    assert not np.allclose(outputSignal, 0)

    # Check if the output signal is of the correct data type
    assert outputSignal.dtype == np.float32