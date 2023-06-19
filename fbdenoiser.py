import torch
import torchaudio
import soundfile as sf
import numpy as np
from denoiser.enhance import *

def enhance_waveform(inputSignal, sr=16000):
    # Load model
    model = pretrained.get_model(args).to(args.device)
    model.eval()

    # Prepare input

    # noisy_signals = torch.from_numpy(inputSignal).unsqueeze(0)
    noisy_signals = np.expand_dims(inputSignal, axis=0)
    noisy_signals = noisy_signals.astype(np.float32)
    noisy_signals = torch.tensor(noisy_signals)

    # Forward
    with torch.no_grad():
        # estimate = get_estimate(model, noisy_signals, args)
        estimate = model(noisy_signals)
        estimate = (1 - args.dry) * estimate + args.dry * noisy_signals

    # Convert output to numpy array
    outputSignal = estimate.squeeze(0).cpu().numpy().squeeze()

    return outputSignal

# Test
# TODO soundfile.py", line 1021, in write assert written == len(data) AssertionError
# but we can still get the output file. 

# np.seterr(divide='ignore', invalid='ignore')

def read_audio_file(file_path):
    audio_data, _ = sf.read(file_path)
    return audio_data

def save_audio_file(signal, file_path, sample_rate=16000):
    sf.write(file_path, signal, sample_rate)

    pretrained.add_model_flags(parser)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--streaming', action="store_true",
                        help="true streaming evaluation for Demucs")


parser = argparse.ArgumentParser(
        'denoiser.enhance',
        description="Speech enhancement using Demucs - Generate enhanced files")
add_flags(parser)
parser.add_argument("--out_dir", type=str, default="enhanced",
                    help="directory putting enhanced wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")

group = parser.add_mutually_exclusive_group()
group.add_argument("--noisy_dir", type=str, default=None,
                   help="directory including noisy wav files")
group.add_argument("--noisy_json", type=str, default=None,
                   help="json file including noisy wav files")


args = parser.parse_args()
logging.basicConfig(stream=sys.stderr, level=args.verbose)
logger.debug(args)

file_path = './CON_T_0010584_reverb.wav'
inputSignal = read_audio_file(file_path).astype(np.double)
print(len(inputSignal))
outputSignal = enhance_waveform(inputSignal)
print(len(outputSignal))