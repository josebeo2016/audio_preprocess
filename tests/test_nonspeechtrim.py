from audio_preprocess import NonspeechTrim
import librosa
import os
import torch
import soundfile as sf
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLE_WAV = os.path.join(BASE_DIR,"data/LA_T_5260333_vocals.wav")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def test_nonspeechtrim():
    npt = NonspeechTrim()
    input_signal, sr = librosa.load(SAMPLE_WAV, sr=16000)
    output_signal = npt.process(input_signal)
    print(output_signal.size)
    sf.write(os.path.join(BASE_DIR,"data/LA_T_5260333_trimmed.wav"), output_signal, sr) # manually check
    assert abs(output_signal.size - len(input_signal)) > 128
    

