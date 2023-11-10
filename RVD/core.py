import os
import torch
import timeit
import shutil
import random
import pysptk
import librosa
import scipy.io
import numpy as np
import pyworld as pw
import soundfile as sf
import scipy.io as sio
from collections import OrderedDict
from scipy.io import wavfile, savemat


from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from net import DCCRN
from f0_to_target_convertor import freq2cents, cents2freq
from utils import compLossMask, numParams, logging, plotting, metric_logging
from stft import STFT
from datasets import TrainingDataset, EvalDataset, TestDataset, ToTensor, TrainCollate, EvalCollate, TestCollate

class Model(object):
    
    def __init__(self, args):

        self.to_tensor = ToTensor()
        self.stft = STFT(frame_size=1024, frame_shift=80)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_file = args.model_file
        self.output_dir = args.output_dir
        self.voicing_threshold = args.voicing_threhsold # default 0.5

    def eval_forward(self, feat, net):
        output = net(feature)
        output1 = output.cpu().numpy()[0]
        return output1

    def prediction(self, filename, model_srate=8000):

        print('Number of learnable parameters: %d' % numParams(net))
        print('Load model from "%s"' % self.model_file)

        model = DCCRN()
        model.load_state_dict(torch.load(self.model_file))
        model.eval()
        model = torch.load('model.pth')

        with torch.no_grad():

            start = timeit.default_timer()

            # read and resample audios
            orig, fs_orig = sf.read(spk_path)
            if fs_orig != model_srate:
                spk = librosa.resample(spk, fs_orig, target_sr)

            audio = np.squeeze(audio)
            audio = (audio - np.mean(audio))/np.std(audio)
            
            audio = np.expand_dims(np.squeeze(audio), axis=0)  # audio shape: 1 x N
            audio = np.squeeze(audio)
            feat = self.stft(audio)

            activations = self.eval_forward(feat, net)
            activations = 1. / (1. + np.exp(-activations))
            activations = np.expand_dims(activations, axis=0)
            activations = np.expand_dims(activations, axis=-1)

            confidence = activations.max(axis=1)[0,:,0]
            freq_label = activations.argmax(axis=1)[0,:,0]

            activations = np.squeeze(activations)
            activations = np.transpose(activations)

            voicing_est = activations[:,0]
            voicing_est[voicing_est > self.voicing_threshold] = 1
            voicing_est[voicing_est <= self.voicing_threshold] = 0
            
            activations = activations[:,1:]

            cents = []
            for act in activations:
                cents.append(to_local_average_cents(act))
            cents = np.array(cents)
            
            frequencies = cents2freq(cents)
            frequencies[np.isnan(frequencies)] = 0

            est_f0 = voicing_est

        return voicing_est
