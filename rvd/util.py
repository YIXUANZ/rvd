import torch
import scipy
import numpy as np

class ToTensor(object):
    r"""Convert ndarrays in sample to Tensors."""
    def __call__(self, x):
        return torch.from_numpy(x).float()

def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num

class SignalToFrames:
    """Chunks a signal into frames"""
    def __init__(self, frame_size=256, frame_shift=80):
        # default window 128 ms, frame shift 10 ms
        self.frame_size = frame_size
        self.frame_shift = frame_shift
    def __call__(self, in_sig):
        frame_size = self.frame_size
        frame_shift = self.frame_shift
        sig_len = in_sig.shape[-1]
        nframes = (sig_len // self.frame_shift) 
        a = np.zeros(list(in_sig.shape[:-1]) + [nframes, self.frame_size])
        start = -frame_size // 2
        end = start + self.frame_size
        for i in range(nframes):
            if end < sig_len:
                if start >= 0:
                    a[..., i, :]=in_sig[..., start:end]
                else:
                    head_pos = -start
                    a[..., i, head_pos:]=in_sig[...,:end]
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size]=in_sig[..., start:]
            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class STFT:
    """Computes STFT of a signal"""
    def __init__(self, frame_size=256, frame_shift=80):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = scipy.hamming(frame_size)
        self.get_frames = SignalToFrames(self.frame_size, self.frame_shift)
    def __call__(self, signal):
        frames = self.get_frames(signal)
        frames = frames*self.win
        feature = np.fft.fft(frames)[..., 0:(self.frame_size//2+1)]
        feat_R = np.real(feature)
        feat_I = np.imag(feature)
        feature = np.stack([feat_R, feat_I], axis=0)
        return feature

