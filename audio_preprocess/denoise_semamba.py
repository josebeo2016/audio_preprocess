# Audio Denoising
# Author: Long Nguyen-Vu & ChatGPT, code adapted from https://github.com/AP-Atul/Audio-Denoising/tree/master
# Date: 2023-06-15

import pywt
import librosa
from tqdm import tqdm
import soundfile as sf
import numpy as np
import glob
import os
import sys
import argparse
import json
import torch


# make script cd to other directory to import the module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the SEMamba directory to sys.path
SEMAMBA_DIR = os.path.join(CURRENT_DIR, 'SEMamba')
sys.path.insert(0, SEMAMBA_DIR)

from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.pcs400 import cal_pcs
import soundfile as sf

from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

class DeNoise_SEMamba:
    """
    process the input signal to remove noise
    inputSignal: np.ndarray
    outputSignal: np.ndarray
    sample_rate: 16000
    """
    def __init__(self):

        # Construct the full paths to the config and checkpoint files
        config_path = os.path.join(SEMAMBA_DIR, 'recipes', 'SEMamba_advanced', 'SEMamba_advanced.yaml')
        checkpoint_path = os.path.join(SEMAMBA_DIR, 'ckpts', 'SEMamba_advanced.pth')

        self.cfg = load_config(config_path)
    
        self.model = SEMamba(self.cfg).to('cuda')
        state_dict = torch.load(checkpoint_path, map_location='cuda')
        self.model.load_state_dict(state_dict['generator'])
        self.model.eval()
  
    def process(self, inputSignal, sr=16000) -> np.ndarray:
        
        n_fft, hop_size, win_size = self.cfg['stft_cfg']['n_fft'], self.cfg['stft_cfg']['hop_size'], self.cfg['stft_cfg']['win_size']
        compress_factor = self.cfg['model_cfg']['compress_factor']
        sampling_rate = self.cfg['stft_cfg']['sampling_rate']
        
        duration = librosa.get_duration(y=inputSignal, sr=sr)
        output_signal = []

        with torch.no_grad():
            noisy_wav = torch.FloatTensor(inputSignal).to('cuda')

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to('cuda')
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
            amp_g, pha_g, com_g = self.model(noisy_amp, noisy_pha)
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_g = audio_g / norm_factor

        return audio_g.cpu().numpy().squeeze()
    
"""
# Test
# TODO soundfile.py", line 1021, in write assert written == len(data) AssertionError
# but we can still get the output file. 

# np.seterr(divide='ignore', invalid='ignore')

def read_audio_file(file_path):
    audio_data, _ = sf.read(file_path)
    return audio_data

def save_audio_file(signal, file_path, sample_rate=16000):
    sf.write(file_path, signal, sample_rate)

file_path = './vinyl.wav'
inputSignal = read_audio_file(file_path)
print(len(inputSignal))
deNoiser = DeNoise(inputSignal)
outputSignal = deNoiser.process()
print(len(outputSignal))

save_audio_file(outputSignal, file_path='./test.test/vinyl_denoised.flac')
"""
