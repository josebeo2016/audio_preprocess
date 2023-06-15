#@title Useful functions, don't forget to execute
import io
from pathlib import Path
import select
from shutil import rmtree
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
import librosa
import numpy as np
import soundfile as sf
import demucs.separate
from demucs import apply 
# import apply_model, BagOfModels
from demucs import audio 
# AudioFile, convert_audio, save_audio
from demucs import htdemucs
# HTDemucs
from demucs import pretrained
# get_model, add_model_flags, ModelLoadingError
import torch
import torchaudio as ta
class SpeechSeperate():
    def __init__(self, config) -> None:
        """
        Require config:
        device: cpu or cuda
        """
        # get model
        self.model = pretrained.get_model(name="htdemucs")
        self.device = config['device']
        
    def process(self, input_signal: np.ndarray, sr=16000) -> np.ndarray:
        print(self.model.sources)

        input_signal_tensor = torch.from_numpy(input_signal).unsqueeze(0).to(self.device)
        # convert to same channel with model: htdemucs use 2 channels
        input_signal_tensor = audio.convert_audio(input_signal_tensor, sr, self.model.samplerate ,self.model.audio_channels)

        # apply model
        output_signal_tensor = apply.apply_model(self.model, input_signal_tensor[None], device=self.device)[0]

        # convert to same channel with input signal
        output_signal_tensor = audio.convert_audio(output_signal_tensor, self.model.samplerate, sr, 1)
        # convert to numpy array
        output_signal_tensor = output_signal_tensor[-1].squeeze(0).cpu().numpy() # the last one is vocals
        
        return output_signal_tensor
    
    