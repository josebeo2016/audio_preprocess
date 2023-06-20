import torch
import numpy as np
from denoiser.enhance import *

def enhance_waveform(inputSignal, sr=16000):
    # Load model
    # model = pretrained.get_model(args).to(args.device)
    model = pretrained.dns48(pretrained=True).to("cpu")   
    model.eval()
    inputSignal = np.array(inputSignal, dtype=np.float32)
    noisy_signals = torch.from_numpy(inputSignal).unsqueeze(0)

    # Forward
    with torch.no_grad():
        estimate = model(noisy_signals)
        # estimate = (1 - args.dry) * estimate + args.dry * noisy_signals

    # Convert output to numpy array
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
# outputSignal = enhance_waveform(inputSignal)
# print(len(outputSignal))
# save_audio_file(outputSignal, './output.wav')