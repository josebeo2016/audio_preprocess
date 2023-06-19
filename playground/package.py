from audio_preprocess import SpeechSeperate
import librosa
import os
import torch
import soundfile as sf
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLE_WAV = os.path.join(BASE_DIR,"LA_T_5260333.wav")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def test_speechseperate():
    ssp = SpeechSeperate(
        config={
            "device": DEVICE
        })
    input_signal, sr = librosa.load(SAMPLE_WAV, sr=16000)
    output_signal = ssp.process(input_signal)
    sf.write(os.path.join(BASE_DIR,"LA_T_5260333_vocals.wav"), output_signal, sr) # manually check
    assert abs(output_signal.size - len(input_signal)) < 5
    
test_speechseperate()