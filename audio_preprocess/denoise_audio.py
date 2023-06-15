import os
import sys
import argparse

sys.path.append(os.path.abspath("../utils/Audio_Denoising"))
from denoise import *


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Audio Denoising')

# Add the command-line arguments
parser.add_argument('--inDir', type=str, help='Input directory path')
parser.add_argument('--outDir', type=str, help='Output directory path')

# Parse the arguments
args = parser.parse_args()

# Assign the input and output directories from the parsed arguments
inDirectory = args.inDir
outDirectory = args.outDir

# Iterate over files in the input directory
for filename in os.listdir(inDirectory):
    if not filename.endswith(".flac"):
        print("not a .flac file")
        continue
    fi = os.path.join(inDirectory, filename)
    fo = os.path.join(outDirectory, filename)

    # Checking if it is a file
    if os.path.isfile(fi):
        print(fi)
    else:
        print("file not found {}".format(fi))
        continue

    audioDenoiser = AudioDeNoise(inputFile=fi)
    audioDenoiser.deNoise(outputFile=fo)
    
    #audioDenoiser.generateNoiseProfile(noiseFile="input_noise_profile.wav")
