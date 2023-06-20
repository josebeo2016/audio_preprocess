import torch
# import torchaudio
import soundfile as sf
import numpy as np
from denoiser.enhance import *

class DeReverb:
    def __init__(self):
        self.model = pretrained.dns48(pretrained=True).to("cpu")
        self.model.eval()

    def process(self, inputSignal, sr=16000):
        inputSignal = np.array(inputSignal, dtype=np.float32)
        noisy_signals = torch.from_numpy(inputSignal).unsqueeze(0)

        with torch.no_grad():
            estimate = self.model(noisy_signals)

        outputSignal = estimate.squeeze(0).cpu().numpy().squeeze()
        return outputSignal





# # Test

# def read_audio_file(file_path):
#     audio_data, _ = sf.read(file_path)
#     return audio_data

# def save_audio_file(signal, file_path, sample_rate=16000):
#     sf.write(file_path, signal, sample_rate)



# file_path = './CON_T_0010584_reverb.wav'
# inputSignal = read_audio_file(file_path)
# print(len(inputSignal))

# denoiser = FBdenoiser()
# outputSignal = denoiser.process(inputSignal)
# print(len(outputSignal))
# save_audio_file(outputSignal, './output.wav')