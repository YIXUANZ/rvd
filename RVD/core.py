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

from dccrn import Net as DCCRN
from f0_to_target_convertor import freq2cents, cents2freq
from utils import compLossMask, numParams, logging, plotting, metric_logging
from stft import STFT
from datasets import TrainingDataset, EvalDataset, TestDataset, ToTensor, TrainCollate, EvalCollate, TestCollate


def voicing_decision_err(grd, est):
    # grd: ground truth voicing status vector
    # est: estimated voicing status vector
    grd[grd > 0] = 1
    est[est > 0] = 1
    count = 0
    total_frames = len(grd)
    for i in range(total_frames):
        if grd[i] != est[i]:
            count += 1
    vde = count/total_frames
    return vde   

def voicing_decision_err_vuv_uvv(grd, est):
    # grd: ground truth voicing status vector
    # est: estimated voicing status vector
    grd[grd > 0] = 1
    est[est > 0] = 1
    vuv = 0
    uvv = 0
    total_frames = len(grd)
    for i in range(total_frames):
        if grd[i] == 1 and est[i] == 0:
            vuv += 1
        elif grd[i] == 0 and est[i] == 1: 
            uvv += 1

    vde_vuv = vuv/total_frames
    vde_uvv = uvv/total_frames
    return vde_vuv, vde_uvv 

class Checkpoint(object):
    def __init__(self, start_epoch=None, start_iter=None, train_loss=None, eval_loss=None, best_loss=np.inf, state_dict=None, optimizer=None):
        self.start_epoch = start_epoch
        self.start_iter = start_iter
        self.train_loss = train_loss
        self.eval_loss = eval_loss
        self.best_loss = best_loss
        self.state_dict = state_dict
        self.optimizer = optimizer
    
    
    def save(self, is_best, filename, best_model):
        print('Saving checkpoint at "%s"' % filename)
        torch.save(self, filename)
        if is_best:
            print('Saving the best model at "%s"' % best_model)
            shutil.copyfile(filename, best_model)
        print('\n')


    def load(self, filename):
        if os.path.isfile(filename):
            print('Loading checkpoint from "%s"\n' % filename)
            checkpoint = torch.load(filename, map_location='cpu')
            
            self.start_epoch = checkpoint.start_epoch
            self.start_iter = checkpoint.start_iter
            self.train_loss = checkpoint.train_loss
            self.eval_loss = checkpoint.eval_loss
            self.best_loss = checkpoint.best_loss
            self.state_dict = checkpoint.state_dict
            self.optimizer = checkpoint.optimizer
        else:
            raise ValueError('No checkpoint found at "%s"' % filename)


class Model(object):
    
    def __init__(self, args):
        # Hyper parameters

        self.input_size = args.input_size 
        self.label_size = args.label_size
        self.batch_size = args.batch_size# 1
        self.model_name = args.model_name
        
        # self.train_norm_param = scipy.io.loadmat('resource/norm_para_tr.mat', mat_dtype=True)
        
        self.to_tensor = ToTensor()
        self.stft = STFT(frame_size=1024, frame_shift=80)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def eval_forward(self, feat, net):
        feature = np.expand_dims(feat, axis=0)
        feature = self.to_tensor(feature)
        feature = feature.to(self.device)
        output = net(feature)
        output1 = output.cpu().numpy()[0] # [0][0]
        del output, feature
        return output1

    def test(self, args, model_srate=8000):
        with open(args.test_list,'r') as test_list_file:
            self.test_list = [line.strip() for line in test_list_file.readlines()]
        self.model_name = args.model_name
        self.model_file = args.model_file
        self.output_dir = args.output_dir
        self.voicing_threshold = 0.5
        
        # create a network
        print('model', self.model_name)
        net = DCCRN()
        net.to(self.device)

        print('Number of learnable parameters: %d' % numParams(net))
        print(net)

        # loss and optimizer
        net.eval()
        print('Load model from "%s"' % self.model_file)
        checkpoint = Checkpoint()
        checkpoint.load(self.model_file)
        state_dict = checkpoint.state_dict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

        with torch.no_grad():

            testSet = TestDataset(self.test_list)

            # create a data loader for test
            test_loader = DataLoader(testSet,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=2,
                                     collate_fn=TestCollate())

            cnt = 0.
            check_file_list = []

            for k, (audio, feat, ref_f0) in enumerate(test_loader):

                start = timeit.default_timer()
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
