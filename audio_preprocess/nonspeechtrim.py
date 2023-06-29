import librosa
import numpy as np
import torch
class NonspeechTrim():
    def __init__(self, config=None) -> None:
        """
        Require config:
        None
        """
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

        (self.get_speech_timestamps,
         _,
         _,
        self.VADIterator,
        self.collect_chunks) = utils
        self.config = config
        
    def process(self, input_signal: np.ndarray, sr=16000) -> np.ndarray:
        # convert to tensor
        input_signal = torch.from_numpy(input_signal)
        speech_timestamps = self.get_speech_timestamps(input_signal, self.model, sampling_rate=sr)
        try:
            trimed_audio = self.collect_chunks(speech_timestamps, input_signal)
            output_signal = trimed_audio.numpy()
        except Exception as e:
            raise e
        
        return output_signal