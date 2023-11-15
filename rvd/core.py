import torch
import librosa
import numpy as np
import soundfile as sf

from .net import Net
from .util import numParams, ToTensor, STFT

class Model(object):
    
    def __init__(self, model_file='./pretrained/rvd_all_weights.pth', voicing_threshold=0.5, device=None):
        '''
        :param args
        '''
        self.to_tensor = ToTensor()
        self.stft = STFT(frame_size=1024, frame_shift=80)
        if device:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_file = model_file
        self.voicing_threshold = voicing_threshold # default 0.5

    def eval_forward(self, feat, net):
        '''
        :param feat
        :param net
        '''
        output = net(feat)
        output1 = output.cpu().numpy()[0]
        return output1

    def predict(self, filename, model_srate=8000):
        '''
        :param filename
        '''
        model = Net()
        print('Number of learnable parameters: %d' % numParams(model))
        print('Load model from "%s"' % self.model_file)

        model.load_state_dict(torch.load(self.model_file))
        model.eval()

        with torch.no_grad():
            # read and resample audios
            speech, fs_orig = sf.read(filename)
            if fs_orig != model_srate:
                speech = librosa.resample(speech, fs_orig, model_srate)

            speech = (speech - np.mean(speech))/np.std(speech)
            feat = self.stft(speech)
            feat = self.to_tensor(feat[None, :])

            activations = self.eval_forward(feat, model)
            voicing_est = 1. / (1. + np.exp(-activations))

            voicing_est[voicing_est > self.voicing_threshold] = 1
            voicing_est[voicing_est <= self.voicing_threshold] = 0

        return voicing_est
