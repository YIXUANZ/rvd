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
import torchcrepe

from dccrn import Net as DCCRN
from f0_to_target_convertor import freq2cents, cents2freq
from utils import compLossMask, numParams, logging, plotting, metric_logging
from stft import STFT
from f0_to_target_convertor import f0_to_target_vector, f0_to_target_vector_multi
from datasets import TrainingDataset, EvalDataset, TestDataset, ToTensor, TrainCollate, EvalCollate, TestCollate


def to_local_average_cents(salience, center=None, fmin=30., fmax=1000., vecSize=486):
    '''
    find the weighted average cents near the argmax bin in output pitch class vector

    :param salience: output vector of salience for each pitch class
    :param fmin: minimum ouput frequency (corresponding to the 1st pitch class in output vector)
    :param fmax: maximum ouput frequency (corresponding to the last pitch class in output vector)
    :param vecSize: number of pitch classes in output vector
    :return: predicted pitch in cents
    '''

    if not hasattr(to_local_average_cents, 'mapping'):
        # the bin number-to-cents mapping
        fmin_cents = freq2cents(fmin)
        fmax_cents = freq2cents(fmax)
        to_local_average_cents.mapping = np.linspace(fmin_cents, fmax_cents, vecSize) # cents values corresponding to the bins of the output vector

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience)) # index of maximum value in output vector
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def detection_rate(grd, est, rt=0.05):
    count = 0
    length = 0
    for i in range(len(grd)):
        if grd[i] > 0:
            if est[i] > 0:
                if abs(grd[i] - est[i]) < grd[i] * rt:
                    count = count + 1
            length = length + 1

    dtr = count/length
    return dtr

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

    def train(self,args):

        self.max_epoch = args.max_epoch #30
        self.eval_steps = args.eval_steps # 500
        self.learning_rate = args.learning_rate # 0.0001
        self.voice_w = args.voice_w
        self.resume = args.resume


        with open(args.train_list,'r') as train_list_file:
            self.train_list = [line.strip() for line in train_list_file.readlines()]
        with open(args.eval_list,'r') as eval_list_file:
            self.cv_list = [line.strip() for line in eval_list_file.readlines()]
        self.num_train_sentences = len(self.train_list)

        self.log_path = args.log_path 
        self.model_path = args.model_path

        # create a training dataset and an evaluation dataset
        # trainSet = TrainingDataset(self.train_list, self.train_norm_param)
        # evalSet = EvalDataset(self.cv_list, self.train_norm_param)
        trainSet = TrainingDataset(self.train_list)
        evalSet = EvalDataset(self.cv_list)

        # create data loaders for training and evaluation
        train_loader = DataLoader(trainSet,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=1, #16
                                  pin_memory=True, 
                                  drop_last=True,
                                  collate_fn=TrainCollate())
        eval_loader = DataLoader(evalSet,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True, 
                                 drop_last=True,
                                 collate_fn=EvalCollate())
        
        # create a network
        print('model', self.model_name)
        os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
        self.dccrn = DCCRN().cuda()
        net = torch.nn.DataParallel(self.dccrn)
        net.to(self.device)
        print('Number of learnable parameters: %d' % numParams(net))
        print(net)


        if self.resume:
            print('Load model from "%s"' % self.resume)
            checkpoint = torch.load(self.resume, map_location='cpu')
            net.load_state_dict(checkpoint.state_dict)
            start_epoch = checkpoint.start_epoch
            start_iter = 0
            best_loss = np.inf
        else:
            start_epoch = 0
            start_iter = 0
            best_loss = np.inf
            # training criterion and optimizer for DCCRN_SP
            
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        num_train_batches = self.num_train_sentences // self.batch_size

        cnt = 0.
        print(best_loss)

        torch.manual_seed(0)

        for epoch in range(start_epoch, self.max_epoch):
            accu_train_loss = 0.0
            net.train()

            start  = timeit.default_timer()
            for i, (features, labels, nframes) in enumerate(train_loader):

                i += start_iter
                # features = torch.unsqueeze(features, axis=1)

                features = features.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32)
                nframes = nframes.to(self.device, dtype=torch.long)

                # forward + backward + optimize
                optimizer.zero_grad()

                # without sequence modeling
                outputs = net(features)
                loss, pitch_loss = self.dccrn.loss(outputs, labels, nframes, self.voice_w)

                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 5.0)
                optimizer.step()

                running_loss = loss.data.item()
                tracking_loss = pitch_loss.data.item()
                accu_train_loss += running_loss

                cnt += 1.

                del loss, outputs, features, labels

                end = timeit.default_timer()

                curr_time = end - start

                print('iter = {}/{}, epoch = {}/{}, loss = {:.5f}, f0_loss = {:.5f}, time/batch = {:.5f}'.format(i+1,
                num_train_batches, epoch+1, self.max_epoch, running_loss, tracking_loss, curr_time))

                if (i+1) % self.eval_steps == 0:

                    start = timeit.default_timer()

                    avg_train_loss = accu_train_loss / cnt

                    avg_eval_loss, avg_eval_acc = self.validate(net, eval_loader)

                    # scheduler.step(avg_eval_loss)

                    net.train()

                    print('Epoch [%d/%d], Iter [%d/%d]  ( TrainLoss: %.4f | EvalLoss: %.4f | EvalAcc: %.4f )' % (epoch+1,self.max_epoch,i+1,self.num_train_sentences//self.batch_size,avg_train_loss,avg_eval_loss,avg_eval_acc))

                    is_best = True if avg_eval_loss < best_loss else False
                    best_loss = avg_eval_loss if is_best else best_loss

                    checkpoint = Checkpoint(epoch, i, avg_train_loss, avg_eval_loss, best_loss, net.state_dict(), optimizer.state_dict())

                    model_name = self.model_name + '_latest.model'
                    best_model = self.model_name + '_best.model'

                    checkpoint.save(is_best, os.path.join(self.model_path, model_name), os.path.join(self.model_path, best_model))

                    logging(self.log_path, self.model_name +'_loss_log.txt', checkpoint, self.eval_steps)

                    accu_train_loss = 0.0
                    cnt = 0.

                    net.train()

                if (i+1)%num_train_batches == 0:
                    break

            avg_eval_loss, avg_eval_acc = self.validate(net, eval_loader)
            scheduler.step(avg_eval_loss)
            net.train()

            print('After {} epoch the performance on validation is:'.format(epoch+1))
            print(avg_eval_loss)

            # checkpoint = Checkpoint(epoch, 0, None, None, best_loss, net.state_dict(), optimizer.state_dict())
            # checkpoint.save(False, os.path.join(self.model_path, self.model_name + '-{}.model'.format(epoch+1)),
            #                 os.path.join(self.model_path, best_model))
            # checkpoint.save(False, os.path.join(self.model_path, self.model_name + '-{}.model'.format('latest')),
            #                 os.path.join(self.model_path, best_model))
            metric_logging(self.log_path, self.model_name +'_metric_log.txt', epoch, [avg_eval_loss, avg_eval_acc])
            start_iter = 0.

    def validate(self, net, eval_loader):
        #print('********** Started evaluation on validation set ********')
        net.eval()
        
        with torch.no_grad():
            mtime = 0
            ttime = 0.
            cnt = 0.
            accu_eval_loss = 0.0
            accu_eval_acc = 0.0
            for k, (feat, label) in enumerate(eval_loader):

                start = timeit.default_timer()
                output = self.eval_forward(feat, net)

                output_exp = self.to_tensor(np.expand_dims(output, axis=0))
                label_exp = self.to_tensor(np.expand_dims(label, axis=0))

                output_exp = output_exp.to(self.device, dtype=torch.float32)
                label_exp = label_exp.to(self.device, dtype=torch.float32)

                eval_loss, f0_loss = self.dccrn.loss(output_exp, label_exp, None, self.voice_w)
                # eval_loss = net.loss(output_exp, label_exp, [label.shape[0]])
                # eval_loss = self.cross_entropy_loss(output, label)
                accu_eval_loss += eval_loss

                accu_eval_acc += f0_loss
                
                cnt += 1.
                
                end = timeit.default_timer()
                curr_time = end - start
                
            avg_eval_loss = accu_eval_loss / cnt
            avg_eval_acc = accu_eval_acc / cnt
        net.train()
        return avg_eval_loss, avg_eval_acc

    def eval_forward(self, feat, net):
        feature = np.expand_dims(feat, axis=0)
        feature = self.to_tensor(feature)
        feature = feature.to(self.device)
        output = net(feature)
        output1 = output.cpu().numpy()[0] # [0][0]
        del output, feature
        return output1

    def cross_entropy_loss(self, pred, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions. 
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray        
        Returns: scalar
        """
        pred = np.clip(pred, epsilon, 1. - epsilon)
        N = pred.shape[0]
        ce = -np.sum(targets*np.log(pred+1e-9))/N
        return ce

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
            ttime = 0.
            mtime = 0.
            accu_test_dr = 0
            accu_test_vde = 0
            accu_test_rapt_vde = 0
            accu_test_loss = 0.0
            accu_test_nframes = 0
            dtr_list = []
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
                rapt_est_f0 = pysptk.sptk.rapt(audio/max(abs(audio))*np.power(2,15), 8000, 80, min=60, max=404, voice_bias=0.0, otype='f0')

                ref_f0 = ref_f0[0,:]
                est_f0 = est_f0[:]
                min_len = min(len(ref_f0), len(est_f0))
                ref_f0 = ref_f0[:min_len]
                est_f0 = est_f0[:min_len]
                rapt_est_f0 = rapt_est_f0[:min_len]

                # if min(ref_f0[ref_f0!=0]) < 30 or max(ref_f0) > 1000:
                #     print("Out of bound!!")
                #     continue

                dtr = 0
                vde = voicing_decision_err(np.copy(ref_f0), np.copy(est_f0))
                rapt_vde = voicing_decision_err(np.copy(ref_f0), np.copy(rapt_est_f0))

                if rapt_vde>0.18:
                    print(k)
                    continue
                # if k == 1:
                #     scipy.io.savemat('fda.mat', {'audio':audio, 'ref_f0': ref_f0, 'est_f0': est_f0, 'rapt_est_f0': rapt_est_f0})
                #     break

                accu_test_dr += dtr
                accu_test_vde += vde
                accu_test_rapt_vde += rapt_vde
                dtr_list.append(dtr)

                cnt += 1
                end = timeit.default_timer()
                curr_time = end - start
                ttime += curr_time
                mtime = ttime / cnt
                mtime = (mtime * (k) + (end-start)) / (k+1)
                print('{}/{}, voicing_decision_error = {:.4f}, rapt_voicing_decision_error = {:.4f}, time/utterance = {:.4f}, '
                        'mtime/utternace = {:.4f}'.format(k+1, len(self.test_list), vde, rapt_vde, curr_time, mtime))
        
        avg_test_dr = accu_test_dr / cnt
        avg_test_vde = accu_test_vde / cnt
        avg_test_rapt_vde = accu_test_rapt_vde / cnt
        std = np.std(np.asarray(dtr_list))
        print('Average detection rate: ', avg_test_dr)
        print('Average voicing decision error: ', avg_test_vde)
        print('Average RAPT voicing decision error: ', avg_test_rapt_vde)
        print("Standard deviation: ", std)
        print(check_file_list)
        print(len(check_file_list))
        # scipy.io.savemat('dr.mat', {'dtr_list': dtr_list})
        end1 = timeit.default_timer()
        print('********** Finisehe working on test files. time taken = {:.4f} **********'.format(end1 - start1))
        dtr_avg_list.append(avg_test_dr)
        vde_avg_list.append(avg_test_vde)
        

    def test_fda(self, args, model_srate=8000, model_input_size=929):

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
        # net.eval()
        # print('Load model from "%s"' % self.model_file)
        # checkpoint = Checkpoint()
        # checkpoint.load(self.model_file)
        # net.load_state_dict(checkpoint.state_dict)
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

        test_set = 'fda'

        if test_set == 'fda':

            # test set
            fda_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/fda/'
            spk = ['rl', 'sb']
            test_list = []

            # noise set
            # test_noise_list = ['/scratch/bbje/zhang23/data/monopitch/test/premix/noise/NoiseX92/babble.wav']
            # test_snr_list = [-10.0]
            test_noise_list = ['/ocean/projects/cis210082p/yixuanz/data/monopitch/test/NoiseX92/babble.wav', '/scratch/bbje/zhang23/data/monopitch/test/NoiseX92/factory2.wav', '/scratch_pnl18/to_YiXuan/ADTcafeteria.wav']
            test_snr_list = [-10.0, -5.0, 0.0, 5.0]
            constant = 1.0 # used for energy normalization

            # Female
            for spk_i in spk:
                spk_list = os.listdir(os.path.join(fda_path, spk_i))
                spk_list.sort()
                test_list = test_list + [fda_path + spk_i + '/' + s for s in spk_list if '.wav' in s]

            with torch.no_grad():

                dtr_avg_list = []
                vde_avg_list = []
                random.seed(1)

                for noise_name in test_noise_list:
                    print('Using %s noise' % (noise_name))
                    # read noise
                    n, srate_n = sf.read(noise_name)
                    n_t = librosa.resample(n, srate_n, model_srate)

                    for snr in test_snr_list:
                        print('SNR level: %s dB' % snr)
                        start1 = timeit.default_timer()
                        accu_test_dr = 0.0
                        accu_test_vde = 0.0
                        accu_test_rapt_vde = 0.0
                        accu_test_nframes = 0
                        ttime = 0.
                        mtime = 0.
                        cnt = 0.
                        dtr_list = []

                        for k, sndFile in enumerate(test_list):
                            
                            (sr, audio) = wavfile.read(sndFile)

                            if len(audio.shape) == 2:
                                audio = audio.mean(1)  # make mono
                            audio = audio.astype(np.float32)

                            if sr != model_srate: # resample audio if necessary
                                from resampy import resample
                                audio = resample(audio, sr, model_srate)

                            s = audio
                            # choose a point where we start to cut
                            start_cut_point = random.randint(0,n.size-s.size)
                            while np.sum(n[start_cut_point:start_cut_point+s.size]**2.0) == 0.0:
                                start_cut_point = random.randint(0,n.size-s.size)

                            # cut noise
                            # n_t = n[start_cut_point:start_cut_point+s.size]                      
                            
                            # cut noise
                            if len(n.shape) > 1:
                                n_t = n[start_cut_point:start_cut_point+s.size, 0]
                            else:
                                n_t = n[start_cut_point:start_cut_point+s.size]    

                            # mixture = speech + noise
                            # alpha = np.sqrt(np.sum(s**2.0)/(np.sum(n_t**2.0)*(10.0**(snr/10.0))))
                            alpha=0
                            snr_check = 10.0*np.log10(np.sum(s**2.0)/np.sum((n_t*alpha)**2.0))
                            try: 
                                mix = s + alpha * n_t
                            except:
                                print(s.shape, n_t.shape)
                                continue
                            # energy normalization
                            c = np.sqrt(constant*mix.size/np.sum(mix**2))
                            mix = mix * c
                            s = s * c

                            audio = mix
                            audio = (audio - np.mean(audio))/np.std(audio)

                            rapt_est_f0 = pysptk.sptk.rapt(audio/max(abs(audio))*np.power(2,15), 8000, 80, min=60, max=404, voice_bias=0.0, otype='f0')


                            audio = np.squeeze(audio)
                            feature = self.stft(audio)
                            # feature_s = self.stft(s)
                            feat = np.swapaxes(feature, 1, 2)
                            # feature = self.stft(audio)
                            # feat = np.sqrt(feature[0, :, :] ** 2 + feature[1, :, :] ** 2).T  
                            # feat = np.expand_dims(feat, 0) #(batch_size, 513, 601)

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

                            # voicing_est = rapt_est_f0
                            # voicing_est[voicing_est>0]=1
                            # est_f0 = frequencies
                            est_f0 = frequencies * voicing_est

                            # Ref f0
                            ref_f0_path_arr = sndFile.split("/")
                            ref_f0_sentence = ref_f0_path_arr[-1].split('_')

                            ref_f0_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/fda/FDA_pitch/cstr_consensus_' + ref_f0_sentence[0][:-4] + '/pitch.npy'
                            ref_lar_f0_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/fda/FDA_pitch/cstr_consensus_' + ref_f0_sentence[0][:-4] + '/lar_f0.npy'
                            ref_time_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/fda/FDA_pitch/cstr_consensus_' + ref_f0_sentence[0][:-4] + '/time.npy'
                            ref_prob_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/fda/FDA_pitch/cstr_consensus_' + ref_f0_sentence[0][:-4] + '/probability.npy'
                            ref_voicing_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/fda/FDA_pitch/cstr_consensus_' + ref_f0_sentence[0][:-4] + '/voicing.npy'


                            if os.path.exists(ref_f0_path):

                                ref_f0_mat = np.load(ref_f0_path)
                                ref_f0_ = ref_f0_mat
                                # ref_lar_f0 = np.load(ref_lar_f0_path)
                                # ref_time = np.load(ref_time_path)

                                ref_voicing = np.load(ref_voicing_path)
                                # ref_voicing[ref_voicing>0] = 1
                                ref_f0 = ref_f0_ * ref_voicing

                                # ref_prob = np.load(ref_prob_path)
                                # ref_voiced = ref_prob
                                # threshold = 0.7
                                # ref_voiced[ref_voiced > threshold] = 1
                                # ref_voiced[ref_voiced <= threshold] = 0
                                # ref_f0 = ref_f0_ * ref_voiced


                                # print(len(ref_f0), len(est_f0))
                                ref_f0 = ref_f0[1:]
                                est_f0 = est_f0[:]
                                min_len = min(len(ref_f0), len(est_f0))
                                ref_f0 = ref_f0[:min_len]
                                est_f0 = est_f0[:min_len]

                                if min(ref_f0[ref_f0!=0]) < 30 or max(ref_f0) > 1000:
                                    print("Out of bound!!")
                                    continue
                                
                                # if k < 50:
                                #     continue

                                dtr = detection_rate(ref_f0, est_f0)
                                vde = voicing_decision_err(np.copy(ref_f0[:]), np.copy(est_f0[:]))
                                rapt_vde = voicing_decision_err(np.copy(ref_f0[:]), np.copy(rapt_est_f0[:]))

                                accu_test_dr += dtr
                                accu_test_vde += vde
                                accu_test_rapt_vde += rapt_vde
                                dtr_list.append(dtr)

                                cnt += 1
                                end = timeit.default_timer()
                                curr_time = end - start
                                ttime += curr_time
                                mtime = ttime / cnt
                                mtime = (mtime * (k) + (end-start)) / (k+1)
                                print(sndFile)
                                print('{}/{}, voicing_decision_error = {:.4f}, rapt_voicing_decision_error = {:.4f}, time/utterance = {:.4f}, '
                                       'mtime/utternace = {:.4f}'.format(k+1, len(test_list), vde, rapt_vde, curr_time, mtime))
                                
                                # if k == 19:
                                #     scipy.io.savemat('fda.mat', {'audio':audio, 'ref_f0': ref_f0, 'est_f0': est_f0, 'rapt_est_f0': rapt_est_f0})
                                #     break
                                # if k+1 == 53:
                                #     scipy.io.savemat('plot_53.mat', {'mixstft':feat, 'ref_f0': ref_f0, 'est_f0': est_f0})
                                #     assert(1==2)
                                # if k+1 == 1:
                                #     scipy.io.savemat('plot_fda.mat', {'mix':audio, 'ref_f0': ref_f0})
                                #     assert(1==2)
                            # scipy.io.savemat(self.output_dir + '/test'+str(k+1)+'.mat', {'est_pitch': frequencies, 'ref_f0': ref_f0})
                        
                        avg_test_dr = accu_test_dr / cnt
                        avg_test_vde = accu_test_vde / cnt
                        avg_test_rapt_vde = accu_test_rapt_vde / cnt
                        std = np.std(np.asarray(dtr_list))
                        print('Average detection rate: ', avg_test_dr)
                        print('Average voicing decision error: ', avg_test_vde)
                        print('Average voicing decision error of RAPT: ', avg_test_rapt_vde)
                        print("Standard deviation: ", std)
                        # scipy.io.savemat('dr.mat', {'dtr_list': dtr_list})
                        end1 = timeit.default_timer()
                        print('********** Finisehe working on test files. time taken = {:.4f} **********'.format(end1 - start1))
                        dtr_avg_list.append(avg_test_dr)
                        vde_avg_list.append(avg_test_vde)
                        assert(1==2)
                
                print(dtr_avg_list, vde_avg_list)


    def test_synth(self, args, model_srate = 8000):

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
        net.load_state_dict(checkpoint.state_dict)

        # test set
        test_file = '../filelists/libri100_syntest_10spk_100utts.lst'
        with open(test_file,'r') as test_list_file:
            test_list = [line.strip() for line in test_list_file.readlines()]

        # # noise set
        test_noise_list = ['/scratch/bbje/zhang23/data/monopitch/test/premix/noise/NoiseX92/babble.wav', '/scratch/bbje/zhang23/data/monopitch/test/premix/noise/NoiseX92/factory2.wav', '/scratch_pnl18/to_YiXuan/ADTcafeteria.wav']
        test_snr_list = [-10.0, -5.0, 0.0, 5.0]
        constant = 1.0 # used for energy normalization


        with torch.no_grad():

            dtr_avg_list = []
            vde_avg_list = []
            random.seed(1)

            for noise_name in test_noise_list:
                print('Using %s noise' % (noise_name))
                # read noise
                n, srate_n = sf.read(noise_name)
                n_t = librosa.resample(n, srate_n, model_srate)

                for snr in test_snr_list:
                    print('SNR level: %s dB' % snr)
                    start1 = timeit.default_timer()
                    accu_test_dtr = 0.0
                    accu_test_vde = 0.0
                    accu_test_nframes = 0
                    ttime = 0.
                    mtime = 0.
                    cnt = 0.
                    dtr_list = []
                    vde_list = []

                    for k, sndFile in enumerate(test_list):

                        mat = sio.loadmat(sndFile)

                        audio = mat['s_feature']
                        ref_f0 = mat['ref_f0']
                        sr = mat['fs']

                        audio = np.squeeze(audio)
                        ref_f0 = np.squeeze(ref_f0)
                        sr = np.squeeze(sr)

                        if len(audio.shape) == 2:
                            audio = audio.mean(1)  # make mono
                        audio = audio.astype(np.float32)

                        if sr != model_srate: # resample audio if necessary
                            from resampy import resample
                            audio = resample(audio, sr, model_srate)

                        s = audio
                        # choose a point where we start to cut
                        start_cut_point = random.randint(0,n.size-s.size)
                        while np.sum(n[start_cut_point:start_cut_point+s.size]**2.0) == 0.0:
                            start_cut_point = random.randint(0,n.size-s.size)

                        # cut noise
                        if len(n.shape) > 1:
                            n_t = n[start_cut_point:start_cut_point+s.size, 0]
                        else:
                            n_t = n[start_cut_point:start_cut_point+s.size]    

                        # mixture = speech + noise
                        # alpha = np.sqrt(np.sum(s**2.0)/(np.sum(n_t**2.0)*(10.0**(snr/10.0))))
                        alpha = 0
                        snr_check = 10.0*np.log10(np.sum(s**2.0)/np.sum((n_t*alpha)**2.0))
                        try: 
                            mix = s + alpha * n_t
                        except:
                            print(s.shape, n_t.shape)
                        # energy normalization
                        c = np.sqrt(constant*mix.size/np.sum(mix**2))
                        mix = mix * c
                        s = s * c
                        
                        audio = np.squeeze(mix)
                        feature = self.stft(audio)
                        feat = np.swapaxes(feature, 1, 2)

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

                        # est_f0 = frequencies
                        est_f0 = frequencies * voicing_est

                        min_len = min(len(ref_f0), len(est_f0))
                        ref_f0 = ref_f0[:min_len]
                        est_f0 = est_f0[:min_len]

                        if min(ref_f0[ref_f0!=0]) < 30 or max(ref_f0) > 1000:
                            print("Out of bound!!")
                            continue

                        dtr = detection_rate(ref_f0, est_f0)
                        vde = voicing_decision_err(np.copy(ref_f0), np.copy(est_f0))
                        # vde, _ = voicing_decision_err_vuv_uvv(np.copy(ref_f0), np.copy(est_f0))

                        accu_test_dtr += dtr
                        accu_test_vde += vde
                        dtr_list.append(dtr)
                        # vde_list.append(vde)

                        cnt += 1
                        end = timeit.default_timer()
                        curr_time = end - start
                        ttime += curr_time
                        mtime = ttime / cnt
                        mtime = (mtime * (k) + (end-start)) / (k+1)
                        print('{}/{}, detection_rate = {:.4f}, voicing_decision_error = {:.4f}, time/utterance = {:.4f}, '
                               'mtime/utternace = {:.4f}'.format(k+1, len(test_list), dtr, vde, curr_time, mtime))
                    
                    # scipy.io.savemat(self.output_dir + '/test'+str(k+1)+'.mat', {'est_pitch': frequencies, 'ref_f0': ref_f0})

                    avg_test_dtr = accu_test_dtr / cnt
                    avg_test_vde = accu_test_vde / cnt
                    print('Average detection rate: ', avg_test_dtr)
                    print('Average voicing decision error: ', avg_test_vde)
                    end1 = timeit.default_timer()
                    print('********** Finisehe working on test files. time taken = {:.4f} **********'.format(end1 - start1))
                    dtr_avg_list.append(avg_test_dtr)
                    vde_avg_list.append(avg_test_vde)
                    
                    # scipy.io.savemat('test_results.mat', {'vde': vde_list})
                    # assert(1==2)
            
            print(dtr_avg_list)
            print(vde_avg_list)

    def test_ptdb(self, args, model_srate=8000, model_input_size=929):

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

            # test set
            PTDB_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/PTDB-TUG/'
            SPK_ID = ['FEMALE/MIC/F01', 'MALE/MIC/M01', 'FEMALE/MIC/F02', 'MALE/MIC/M02', 'FEMALE/MIC/F03', 'MALE/MIC/M03', 'FEMALE/MIC/F04', 'MALE/MIC/M04', 'FEMALE/MIC/F05', 'MALE/MIC/M05', 'FEMALE/MIC/F06', 'MALE/MIC/M06', 'FEMALE/MIC/F07', 'MALE/MIC/M07', 'FEMALE/MIC/F08', 'MALE/MIC/M08', 'FEMALE/MIC/F09', 'MALE/MIC/M09', 'FEMALE/MIC/F10', 'MALE/MIC/M10']
            # SPK_ID = ['FEMALE/MIC/F09', 'MALE/MIC/M09']
            test_list = []
            ptdb_tug_discarded_file = ['M03_sx122.mat', 'M03_si867.mat', 'F08_si1882.mat', 'F05_si1372.mat', 'M03_si840.mat', 'F09_si1982.mat', 'M02_si797.mat', 'M07_sx289.mat', 'M03_si837.mat', 'M04_si1062.mat', 'M07_si1657.mat', 'F02_si807.mat', 'M02_si649.mat', 'F02_si694.mat', 'M02_si774.mat', 'F04_si1032.mat', 'F08_si1849.mat', 'F04_si1124.mat', 'M07_sa2.mat', 'F01_si600.mat', 'M03_si841.mat', 'M08_sx354.mat', 'M02_si780.mat', 'M05_si1375.mat', 'F02_si735.mat', 'F05_si1343.mat', 'F06_si1499.mat', 'M02_si771.mat', 'F04_si1180.mat', 'F04_si1138.mat', 'M08_sx324.mat', 'F10_si2210.mat', 'M06_si1444.mat', 'F03_si869.mat', 'M09_si2066.mat', 'M02_si798.mat', 'M03_sx123.mat', 'F03_si957.mat', 'F04_si1127.mat', 'F08_si1901.mat', 'F06_si1584.mat', 'F06_si1579.mat', 'F05_si1379.mat', 'M06_si1584.mat', 'F01_si485.mat', 'F10_si2320.mat', 'M01_si585.mat', 'F10_si2190.mat', 'F08_si1794.mat', 'M02_si795.mat', 'M01_si511.mat', 'F04_si1095.mat', 'F10_si2275.mat', 'F04_si1145.mat', 'F02_si674.mat', 'M02_si766.mat', 'M06_sx260.mat', 'F04_si1206.mat', 'F01_si511.mat', 'F02_si755.mat', 'F10_si2223.mat', 'F03_si937.mat', 'F08_si1803.mat', 'F05_sx190.mat', 'F10_sx443.mat', 'F02_si824.mat', 'F08_si1783.mat', 'F04_si1058.mat', 'F01_si576.mat', 'F08_si1800.mat', 'M07_si1656.mat', 'F08_si1896.mat', 'M09_si2148.mat', 'F05_si1354.mat', 'M06_si1421.mat', 'F06_si1448.mat', 'F03_sx101.mat', 'F04_si1133.mat', 'F10_si2280.mat', 'F04_si1037.mat', 'F01_si636.mat', 'F08_si1935.mat', 'F01_si605.mat', 'F08_si1874.mat', 'M02_si644.mat', 'M07_sx308.mat', 'F08_si1789.mat', 'M06_si1540.mat', 'F08_si1898.mat', 'M06_si1435.mat', 'F08_si1802.mat', 'M01_si501.mat', 'F04_si1132.mat', 'M04_si1064.mat', 'F04_si1107.mat', 'F02_si737.mat', 'F09_si2046.mat', 'F08_sx338.mat', 'F06_si1502.mat', 'F02_si758.mat', 'M06_si1436.mat', 'F10_si2313.mat', 'F10_sx445.mat', 'F08_si1781.mat', 'F10_si2312.mat', 'F05_si1378.mat', 'F05_si1388.mat', 'F08_si1788.mat', 'M06_si1471.mat', 'F08_sx354.mat', 'F10_si2310.mat', 'F04_si1130.mat', 'M03_si937.mat', 'M06_si1512.mat', 'F08_si1798.mat', 'F08_si1850.mat', 'F04_si1140.mat', 'M07_si1622.mat', 'F04_si1047.mat', 'F04_sx151.mat', 'F05_sx222.mat', 'M03_si858.mat', 'M06_si1470.mat', 'M02_sx60.mat', 'F08_si1931.mat', 'F01_si509.mat', 'F02_si820.mat', 'M02_si801.mat', 'M03_si902.mat', 'M06_si1424.mat', 'F08_si1899.mat', 'F01_si570.mat', 'F04_si1170.mat', 'F08_sa2.mat', 'F10_si2226.mat', 'F01_si613.mat', 'F05_si1279.mat', 'F04_sx156.mat', 'F04_si1144.mat', 'F08_si1799.mat', 'F10_si2285.mat', 'F04_si1050.mat', 'F01_si490.mat', 'M01_si610.mat', 'F06_si1472.mat', 'M06_sx251.mat', 'F04_si1122.mat', 'M02_si800.mat', 'F04_si1031.mat', 'F02_si812.mat', 'F10_si2281.mat', 'F08_sx349.mat', 'F10_si2159.mat', 'M06_si1517.mat', 'M09_si2004.mat', 'F01_si577.mat', 'F04_si1039.mat', 'M09_si2150.mat', 'F04_si1089.mat', 'F06_si1422.mat', 'F04_si1204.mat', 'F04_si1190.mat', 'M03_si975.mat', 'M06_si1439.mat', 'M09_si2135.mat', 'F04_si1142.mat', 'M03_si852.mat', 'M07_si1672.mat', 'F01_si598.mat', 'M09_si2086.mat', 'M03_si1013.mat', 'M01_si532.mat', 'M09_si2070.mat', 'F08_si1785.mat', 'F02_si822.mat', 'F08_si1819.mat', 'F08_si1820.mat', 'F04_sx174.mat', 'F01_si626.mat', 'F09_si2094.mat', 'F04_si1104.mat', 'F06_si1561.mat', 'F10_si2246.mat', 'M02_si678.mat', 'F03_sx100.mat', 'M02_si759.mat', 'F06_si1496.mat', 'F10_si2270.mat', 'F06_si1573.mat', 'F05_si1319.mat', 'F08_si1956.mat', 'M09_si2001.mat', 'M01_si512.mat', 'F08_si1824.mat', 'F03_si895.mat', 'F02_si754.mat', 'F08_sx327.mat', 'F07_si1604.mat', 'F04_si1137.mat', 'M06_sx262.mat', 'F10_si2297.mat', 'M08_si1840.mat', 'M09_sx386.mat', 'F08_si1856.mat', 'M07_si1651.mat', 'M06_si1486.mat', 'F09_si2131.mat', 'F05_si1227.mat', 'F05_si1311.mat', 'M09_sx404.mat', 'F10_si2318.mat', 'F01_si472.mat', 'F08_si1817.mat', 'F08_si1786.mat', 'F04_si1052.mat', 'F10_si2195.mat', 'F08_si1877.mat', 'F08_si1860.mat', 'M03_sx101.mat', 'F04_si1049.mat', 'F01_si634.mat', 'M07_si1598.mat', 'F08_si1904.mat', 'M03_si979.mat', 'F04_sx182.mat', 'F09_sx404.mat', 'M03_sx118.mat', 'M03_si839.mat', 'F10_si2194.mat', 'F07_si1602.mat', 'F08_si1926.mat', 'F04_si1079.mat', 'F08_si1962.mat', 'M07_sx304.mat', 'M04_si1063.mat', 'F08_sx346.mat', 'F05_si1305.mat', 'F08_si1895.mat', 'F03_si960.mat', 'F06_si1519.mat', 'F07_si1771.mat', 'M06_si1485.mat', 'M06_si1423.mat', 'F02_si732.mat', 'F10_si2165.mat', 'F08_sx330.mat', 'F05_sx192.mat', 'M08_si1779.mat', 'M03_si865.mat', 'F02_si796.mat', 'F04_si1136.mat', 'F01_si495.mat', 'M06_si1430.mat', 'M07_sx307.mat', 'F10_si2200.mat', 'M01_si518.mat', 'M02_si812.mat', 'F08_si1960.mat', 'F06_si1447.mat', 'F02_si818.mat', 'F04_si1057.mat', 'F08_si1805.mat', 'F08_si1809.mat', 'F08_si1841.mat', 'F06_sx260.mat', 'F07_si1597.mat', 'M03_si871.mat', 'M03_si847.mat', 'M06_si1577.mat', 'F10_si2167.mat', 'F08_si1890.mat', 'M06_si1505.mat', 'F04_sx178.mat', 'M02_si754.mat', 'F10_si2284.mat', 'M07_sx316.mat', 'M06_si1558.mat', 'F07_si1733.mat', 'M08_si1877.mat', 'F09_si2020.mat', 'M08_sx351.mat', 'F09_sx387.mat', 'F04_si1181.mat', 'M01_si639.mat', 'F01_si611.mat', 'F08_si1933.mat', 'F05_si1374.mat', 'F08_sx335.mat', 'F08_si1853.mat', 'M02_si791.mat', 'M09_si2028.mat', 'F08_si1946.mat', 'F04_si1053.mat', 'M06_si1466.mat', 'M08_si1791.mat', 'F02_si765.mat', 'M09_si2089.mat', 'F02_si734.mat', 'F04_si1121.mat', 'F05_si1391.mat', 'F10_sx431.mat', 'M07_si1654.mat', 'F10_si2193.mat', 'M02_si705.mat', 'M03_si834.mat', 'F05_si1241.mat', 'F10_si2236.mat', 'M07_sx288.mat', 'F05_sx202.mat', 'F10_sx418.mat', 'F04_sx160.mat', 'M02_sx88.mat', 'F08_si1837.mat', 'M07_sx274.mat', 'F03_si997.mat', 'F04_si1033.mat', 'M01_si554.mat', 'M06_si1480.mat', 'F09_si2111.mat', 'F05_si1282.mat', 'F05_si1234.mat', 'M08_si1945.mat', 'F02_sx54.mat', 'F04_si1161.mat', 'M02_si681.mat', 'F01_si479.mat', 'M07_si1588.mat', 'F04_si1110.mat', 'F04_si1191.mat', 'M03_si962.mat', 'F05_si1215.mat', 'M07_sx279.mat', 'F05_si1225.mat', 'F08_si1919.mat', 'F04_si1103.mat', 'M06_si1542.mat', 'F05_si1331.mat', 'M02_si770.mat', 'F08_si1929.mat', 'F06_si1512.mat', 'F10_sx435.mat', 'M07_si1762.mat', 'F01_si630.mat', 'F10_si2198.mat', 'F01_si631.mat', 'M07_sx306.mat', 'F06_si1505.mat', 'F08_si1784.mat', 'F10_si2233.mat', 'F08_si1923.mat', 'M07_si1592.mat', 'F01_si599.mat', 'F08_si1955.mat', 'F08_si1900.mat', 'F08_si1910.mat', 'F04_sx169.mat', 'F04_si1117.mat', 'F10_si2300.mat', 'F04_si1036.mat', 'F04_si1129.mat', 'F01_si510.mat', 'F04_si1178.mat', 'F08_si1836.mat', 'M08_si1913.mat', 'M02_si677.mat', 'M01_si535.mat', 'F10_si2232.mat', 'F05_si1352.mat', 'F08_si1839.mat', 'M07_sx295.mat', 'F06_si1572.mat', 'F08_si1851.mat', 'M02_sx91.mat', 'F08_si1812.mat', 'M07_si1676.mat', 'F04_si1146.mat', 'F02_si811.mat', 'F10_sx426.mat', 'F10_si2288.mat', 'F06_si1449.mat', 'F06_si1501.mat', 'M03_si963.mat', 'F10_si2158.mat', 'F05_sa1.mat', 'F01_si517.mat', 'F10_si2160.mat', 'F05_sx183.mat', 'F10_si2279.mat', 'F08_si1947.mat', 'M04_si1048.mat', 'M08_si1807.mat', 'F05_si1339.mat', 'M03_si992.mat', 'F05_sx186.mat', 'M06_si1426.mat', 'F02_si733.mat', 'F08_si1845.mat', 'F04_si1088.mat', 'F01_si629.mat', 'F04_si1062.mat', 'M02_si726.mat', 'F08_si1911.mat', 'F04_si1114.mat', 'F02_si684.mat', 'F02_si809.mat', 'F08_si1825.mat', 'M03_si868.mat', 'F04_si1152.mat', 'F10_si2220.mat', 'M03_si898.mat', 'F08_si1848.mat', 'M07_sx311.mat', 'M03_si990.mat', 'M07_si1692.mat', 'M06_si1460.mat', 'F10_si2294.mat', 'F02_si819.mat', 'F03_si876.mat', 'F08_si1906.mat', 'F02_si830.mat', 'M07_sx303.mat', 'M02_si676.mat', 'M07_si1630.mat', 'F02_si771.mat', 'M07_si1611.mat', 'M07_si1613.mat', 'F02_si736.mat', 'M07_si1616.mat', 'F04_sx164.mat', 'M02_sx90.mat', 'F08_sa1.mat', 'F08_si1791.mat', 'M06_si1450.mat', 'F02_si702.mat', 'F04_si1195.mat', 'M06_si1472.mat', 'M09_si2090.mat', 'M08_si1952.mat', 'M07_si1760.mat', 'M06_si1462.mat', 'F10_si2243.mat', 'M07_si1590.mat', 'F08_si1792.mat', 'F09_sx379.mat', 'F04_si1097.mat', 'M07_si1602.mat', 'F04_si1201.mat', 'F04_sx172.mat', 'M03_sx131.mat', 'F10_si2251.mat', 'M04_si1039.mat', 'F05_si1258.mat', 'M08_si1964.mat', 'F08_si1843.mat', 'F02_si788.mat', 'F02_si772.mat', 'F08_si1852.mat', 'F05_si1308.mat', 'F01_si530.mat', 'F01_si609.mat', 'M02_si789.mat', 'F04_si1118.mat', 'F08_si1807.mat', 'M09_sx364.mat', 'F08_si1821.mat', 'M09_si2151.mat', 'M01_si623.mat', 'M08_sx349.mat', 'F03_sx121.mat', 'F07_si1588.mat', 'F05_si1373.mat', 'M09_si1994.mat', 'M01_si459.mat', 'M03_si1019.mat', 'M03_sx93.mat', 'F04_si1194.mat', 'M06_si1432.mat', 'M08_si1801.mat', 'F02_si802.mat', 'F08_sx332.mat', 'M02_si755.mat', 'M03_sx137.mat', 'F04_si1020.mat', 'F08_si1832.mat', 'M08_sx348.mat', 'M09_sx383.mat', 'F06_si1441.mat', 'M07_sx310.mat', 'F04_si1048.mat', 'F10_si2231.mat', 'F04_si1055.mat', 'M07_si1605.mat', 'F02_si742.mat', 'M02_si794.mat', 'F08_si1941.mat', 'M09_si1974.mat', 'M09_si2093.mat', 'M02_si668.mat', 'F03_si973.mat', 'M09_si2073.mat', 'F08_si1818.mat', 'M02_si734.mat', 'M03_si883.mat', 'M09_si2103.mat', 'M04_sx160.mat', 'M07_sx276.mat', 'M09_si2009.mat', 'F02_si654.mat', 'F10_si2196.mat', 'F08_si1810.mat', 'F08_sx326.mat', 'F10_si2248.mat', 'F08_sx324.mat', 'M02_si654.mat', 'M05_si1241.mat', 'M08_sx347.mat', 'F05_si1323.mat', 'M07_sa1.mat', 'F04_sx149.mat', 'F08_si1950.mat', 'M09_sx401.mat', 'F10_si2273.mat', 'F04_sx170.mat', 'F02_si660.mat', 'F10_si2222.mat', 'F04_si1076.mat', 'F08_si1830.mat', 'F08_si1797.mat', 'M05_si1300.mat', 'M09_si2033.mat', 'F10_si2192.mat', 'M02_si777.mat', 'M02_si673.mat', 'M06_si1482.mat', 'F01_si498.mat', 'M08_sx336.mat', 'F04_si1207.mat', 'M05_si1359.mat', 'F08_si1804.mat', 'F04_sx177.mat', 'F05_sx188.mat', 'F10_sx408.mat', 'F08_si1869.mat', 'F08_si1878.mat', 'M06_si1552.mat', 'F05_sx187.mat', 'F04_si1021.mat', 'M07_si1618.mat', 'F02_si746.mat', 'M02_si679.mat', 'M01_si615.mat', 'M02_si643.mat', 'M08_si1866.mat', 'F04_si1148.mat', 'F04_si1096.mat', 'F04_sx163.mat', 'F10_sa1.mat', 'M09_si2072.mat', 'M07_si1606.mat', 'M08_sx335.mat', 'M04_si1141.mat', 'F10_si2307.mat', 'F10_si2299.mat', 'M07_sx285.mat', 'M08_sx352.mat', 'F05_si1248.mat', 'F08_si1893.mat', 'F04_si1115.mat', 'F08_si1875.mat', 'F02_si726.mat', 'M06_si1475.mat', 'M07_sx278.mat', 'M03_sx114.mat', 'M01_si540.mat', 'F07_si1775.mat', 'F10_si2295.mat', 'F10_si2314.mat', 'F06_si1565.mat', 'M08_si1928.mat', 'M09_si2139.mat', 'F04_si1154.mat', 'F02_si768.mat', 'F05_si1309.mat', 'F04_si1065.mat', 'M06_si1479.mat', 'F08_si1958.mat', 'M06_si1538.mat', 'F04_si1135.mat', 'M09_si2014.mat', 'F01_si584.mat', 'F02_si715.mat', 'M02_si667.mat', 'F02_si729.mat', 'F10_si2296.mat', 'F10_si2302.mat', 'M07_sx275.mat', 'F04_si1187.mat', 'F05_si1306.mat', 'F02_si804.mat', 'F03_si841.mat', 'M07_sx313.mat', 'F02_si810.mat', 'M09_si2041.mat', 'F08_sx318.mat', 'F02_si728.mat', 'F05_si1315.mat', 'M09_si2048.mat', 'F10_si2303.mat', 'M04_si1201.mat', 'F02_si730.mat', 'F02_si773.mat', 'F03_si977.mat', 'M06_sa2.mat', 'F07_si1720.mat', 'F01_si612.mat', 'F05_si1250.mat', 'F09_si1981.mat', 'F05_sx189.mat', 'F08_si1835.mat', 'F04_si1108.mat', 'M07_si1589.mat', 'F08_si1813.mat', 'F01_si640.mat', 'F04_si1184.mat', 'F08_si1866.mat', 'F02_si759.mat', 'M06_si1508.mat', 'M06_si1457.mat', 'F08_si1838.mat', 'F10_si2255.mat', 'M07_sx314.mat', 'F02_si725.mat', 'F04_sx175.mat', 'F07_si1606.mat', 'F05_sx221.mat', 'F02_si745.mat', 'M07_si1771.mat', 'M05_si1283.mat', 'M09_si1989.mat', 'M06_si1487.mat', 'F01_si621.mat', 'M07_sx290.mat', 'M07_sx283.mat', 'M08_si1890.mat', 'F04_si1066.mat', 'M02_si799.mat', 'F05_si1359.mat', 'F08_si1816.mat', 'F04_si1134.mat', 'F05_si1392.mat', 'M07_si1694.mat', 'F05_sx220.mat', 'M06_si1563.mat', 'M02_si675.mat', 'F10_sx449.mat', 'F04_si1125.mat', 'M07_si1599.mat', 'F04_si1027.mat', 'M02_si769.mat', 'F08_si1920.mat', 'F08_si1834.mat', 'M09_si2118.mat', 'M07_si1674.mat', 'F02_si829.mat', 'M03_si879.mat', 'F02_si741.mat', 'M05_si1395.mat', 'F04_si1160.mat', 'M07_si1597.mat', 'M09_si2143.mat', 'M08_si1843.mat', 'F10_si2204.mat', 'F10_si2271.mat', 'M06_si1576.mat', 'F02_si762.mat', 'F02_si688.mat', 'F05_si1326.mat', 'F08_si1815.mat', 'F02_si797.mat', 'F04_si1159.mat', 'M03_sx120.mat', 'M01_si602.mat', 'F02_si656.mat', 'M07_sx291.mat', 'M09_si2045.mat', 'M07_si1595.mat', 'F10_si2162.mat', 'F10_si2309.mat', 'F05_si1232.mat', 'F10_si2291.mat', 'F08_si1854.mat', 'F05_si1390.mat', 'M03_si903.mat', 'F01_si632.mat', 'F02_si792.mat', 'F08_si1790.mat', 'F03_si932.mat', 'F04_si1069.mat', 'F02_si750.mat', 'F02_si794.mat', 'F04_si1175.mat', 'M08_si1818.mat', 'M03_si1003.mat', 'F04_sx167.mat', 'F01_si507.mat', 'M07_si1733.mat', 'F08_si1840.mat', 'F04_sx173.mat', 'M05_si1252.mat', 'F08_si1831.mat', 'M01_si589.mat', 'M05_si1255.mat', 'M08_si1828.mat', 'F02_si823.mat', 'F04_sx166.mat', 'F10_si2324.mat', 'M09_sx400.mat', 'F05_si1395.mat', 'F08_sx347.mat', 'F10_sx433.mat', 'F02_si749.mat', 'M07_si1587.mat', 'M03_si964.mat', 'F08_si1945.mat', 'M06_si1564.mat', 'F08_si1827.mat', 'M01_si582.mat', 'F07_sx311.mat', 'F05_si1244.mat', 'F10_si2215.mat', 'F08_si1776.mat', 'F08_si1863.mat', 'F04_sx153.mat', 'F10_si2253.mat', 'F04_si1090.mat', 'M01_si591.mat', 'F07_si1610.mat', 'F10_si2301.mat', 'M08_si1803.mat', 'M07_sx296.mat', 'M02_si656.mat', 'M03_si957.mat', 'F10_si2268.mat', 'M08_si1898.mat', 'F08_si1806.mat', 'M05_si1210.mat', 'F08_si1897.mat', 'F10_si2293.mat', 'M02_si665.mat', 'M06_si1404.mat', 'F04_si1141.mat', 'F05_si1210.mat', 'F04_si1043.mat', 'F04_sx147.mat', 'F08_si1828.mat', 'M03_si835.mat', 'F05_si1312.mat', 'M06_si1452.mat', 'M09_si2096.mat', 'F01_si572.mat', 'M07_sx305.mat', 'M09_si1987.mat', 'F04_si1086.mat', 'F08_si1858.mat', 'F01_si639.mat', 'F08_si1793.mat', 'F04_si1080.mat', 'F04_si1131.mat', 'F08_si1801.mat', 'F04_si1185.mat', 'M02_si804.mat', 'F07_si1608.mat', 'F05_si1273.mat', 'M01_si576.mat', 'M02_si793.mat', 'M09_si2102.mat', 'F08_si1961.mat', 'F04_si1192.mat', 'M07_si1591.mat', 'M03_sx115.mat', 'F06_si1521.mat', 'M03_si970.mat', 'F04_si1093.mat', 'M02_si802.mat', 'M09_si2144.mat', 'F02_si764.mat', 'F08_si1782.mat', 'F04_si1205.mat', 'M08_si1875.mat', 'F04_si1111.mat', 'F02_si744.mat', 'M08_si1863.mat', 'M06_si1521.mat', 'M03_si846.mat', 'M07_sx292.mat', 'F02_si710.mat', 'F10_si2316.mat', 'M09_si1990.mat', 'M08_si1938.mat', 'F05_si1230.mat', 'F04_sx171.mat', 'M08_si1883.mat', 'F06_sx232.mat', 'F04_si1165.mat', 'F08_si1855.mat', 'F02_si650.mat', 'F01_si574.mat', 'F09_si2112.mat', 'F10_si2276.mat', 'F04_si1063.mat', 'F05_si1345.mat', 'F04_si1078.mat', 'F02_si712.mat', 'F02_si763.mat', 'M03_si861.mat', 'F04_si1208.mat', 'F08_si1864.mat', 'F10_si2262.mat', 'M09_si2075.mat', 'F08_si1811.mat', 'M09_si1981.mat', 'F01_si637.mat', 'F02_si814.mat', 'F02_si780.mat', 'F04_si1082.mat', 'M02_si803.mat', 'F04_si1186.mat', 'F08_si1823.mat', 'F05_si1349.mat', 'M08_si1858.mat', 'F02_si790.mat', 'F05_si1375.mat', 'F10_si2235.mat', 'F04_sx165.mat', 'M08_si1815.mat', 'F02_si821.mat', 'M07_sx300.mat', 'F05_si1265.mat', 'F07_si1648.mat', 'F03_sx127.mat', 'M07_si1757.mat', 'F10_si2290.mat', 'M03_si886.mat', 'M01_si514.mat', 'M07_si1596.mat', 'F07_sx316.mat', 'F04_si1042.mat', 'M03_si986.mat', 'F10_si2272.mat', 'M02_si732.mat', 'F08_si1921.mat', 'M08_si1793.mat', 'F03_si846.mat', 'F03_si925.mat', 'F08_si1862.mat', 'F08_si1861.mat', 'F08_si1918.mat', 'F04_si1023.mat', 'F10_sx452.mat', 'F02_si731.mat', 'F02_si723.mat', 'F04_si1087.mat', 'F05_si1394.mat', 'F09_si2023.mat', 'M07_si1659.mat', 'F01_si516.mat', 'M06_si1443.mat', 'F02_sx49.mat', 'F08_si1808.mat', 'F02_si761.mat', 'M06_si1463.mat', 'F01_si482.mat', 'M06_si1572.mat', 'F10_si2304.mat', 'F02_si789.mat', 'M07_sx299.mat', 'F04_si1092.mat', 'F04_si1109.mat', 'F09_si2102.mat', 'M07_sx312.mat', 'M03_sx95.mat', 'F08_si1937.mat', 'F10_si2292.mat', 'F08_si1944.mat', 'F05_si1337.mat', 'M08_si1813.mat', 'F04_sx159.mat', 'F02_si816.mat', 'M07_sx297.mat', 'M10_si2196.mat', 'F09_si2063.mat', 'M07_si1648.mat', 'F03_si835.mat', 'F05_sx191.mat', 'M06_si1541.mat', 'M02_si752.mat', 'F04_sx162.mat', 'M07_si1704.mat', 'F10_si2260.mat', 'M07_si1603.mat', 'F09_sx403.mat', 'F03_si929.mat', 'M02_si775.mat', 'M05_si1350.mat', 'F08_si1859.mat', 'F08_si1829.mat', 'F04_si1060.mat', 'F04_si1123.mat', 'M07_si1628.mat', 'F06_si1537.mat', 'F04_si1126.mat', 'F08_sx321.mat', 'F08_si1847.mat', 'F09_si1995.mat', 'M01_si614.mat', 'F10_sx442.mat', 'F10_si2277.mat', 'M08_si1830.mat', 'F08_si1887.mat', 'F02_si770.mat', 'M02_sx92.mat', 'M09_si2099.mat', 'F04_si1100.mat', 'M02_si682.mat', 'M09_si2101.mat', 'M02_si767.mat', 'F02_si701.mat', 'F10_si2217.mat', 'F10_si2256.mat', 'F10_si2249.mat', 'M02_si669.mat', 'F06_si1497.mat', 'F08_si1870.mat', 'F05_si1235.mat', 'F03_sx106.mat', 'M07_si1621.mat', 'M05_si1367.mat', 'M02_si642.mat', 'F08_si1844.mat', 'F10_si2308.mat', 'M02_si757.mat', 'F05_si1330.mat', 'M09_si2039.mat', 'F02_si748.mat', 'M09_si2035.mat', 'M06_sx269.mat', 'M08_si1856.mat', 'F08_si1959.mat', 'F02_si805.mat', 'M04_sx156.mat', 'F10_si2173.mat', 'F04_si1054.mat', 'F01_si575.mat', 'M06_si1507.mat', 'F04_si1106.mat', 'F04_sx168.mat', 'F08_si1873.mat', 'M07_sx293.mat', 'F04_si1143.mat', 'M07_si1631.mat', 'F08_si1892.mat', 'F09_si2057.mat', 'F02_si739.mat', 'F04_si1112.mat', 'M06_si1553.mat', 'F03_si1016.mat', 'M08_si1802.mat', 'F10_sx420.mat', 'F10_sx438.mat', 'F08_si1780.mat', 'M02_si768.mat', 'F05_si1333.mat', 'F08_si1951.mat', 'M08_si1845.mat', 'F04_sx148.mat', 'F04_si1024.mat', 'F10_si2202.mat', 'F08_si1814.mat', 'F10_si2282.mat', 'M06_si1484.mat', 'M03_sx110.mat', 'F04_sx176.mat', 'F04_si1073.mat', 'F04_si1113.mat', 'F04_si1045.mat', 'F04_si1077.mat', 'M06_si1581.mat', 'F04_si1119.mat', 'F04_si1164.mat', 'F08_si1927.mat', 'F07_sx273.mat', 'M02_si773.mat', 'F04_si1183.mat', 'M04_si1050.mat', 'M02_si805.mat', 'F10_si2317.mat', 'F05_si1396.mat', 'M01_si593.mat', 'M08_si1958.mat', 'F06_si1402.mat', 'F04_si1044.mat', 'F01_si597.mat', 'F05_si1322.mat', 'F10_si2228.mat', 'M08_sx323.mat', 'F09_si2028.mat', 'M03_si936.mat', 'F04_sx139.mat', 'F04_si1099.mat', 'F05_si1251.mat', 'M01_si600.mat', 'M07_sx302.mat', 'M02_si776.mat', 'F08_si1924.mat', 'F01_si488.mat', 'F02_si705.mat', 'F04_si1197.mat', 'M01_si564.mat', 'F07_sx314.mat', 'F05_si1245.mat', 'F02_si657.mat', 'F04_si1038.mat', 'F02_si699.mat', 'F04_si1202.mat', 'F06_si1452.mat', 'F04_si1128.mat', 'F01_sa1.mat', 'F06_sx247.mat', 'F01_si497.mat', 'F10_sx412.mat', 'F10_si2254.mat', 'F08_sx331.mat', 'M07_si1608.mat', 'F05_sx184.mat', 'M09_si2062.mat', 'M03_si845.mat', 'M07_si1610.mat', 'M09_si2032.mat', 'M06_si1455.mat', 'M07_si1677.mat', 'F08_sx325.mat', 'M02_si772.mat', 'F04_si1046.mat', 'M06_si1448.mat', 'M02_si806.mat', 'F06_si1404.mat', 'M07_sx309.mat', 'F04_si1041.mat', 'F08_si1857.mat', 'M03_sx121.mat', 'F02_si682.mat', 'M03_si949.mat', 'F08_si1822.mat', 'F04_si1030.mat', 'F03_si845.mat', 'M09_si2088.mat', 'M08_si1909.mat', 'M02_si778.mat', 'M07_si1609.mat', 'M04_si1069.mat', 'F06_si1568.mat', 'F08_si1948.mat', 'F02_si801.mat', 'F10_si2323.mat', 'F01_si491.mat', 'M03_si897.mat', 'M03_si892.mat', 'F04_si1101.mat', 'M08_si1961.mat', 'F10_si2289.mat', 'F03_si921.mat', 'M01_si612.mat', 'F08_si1889.mat', 'F05_si1357.mat', 'F08_si1833.mat', 'M01_si515.mat', 'F04_si1067.mat', 'M06_si1571.mat', 'M08_si1923.mat', 'F03_si858.mat', 'M03_sa1.mat', 'M03_sx105.mat', 'F03_si969.mat', 'F06_si1511.mat', 'M05_si1221.mat', 'F04_si1071.mat', 'M03_si831.mat', 'F09_si2061.mat', 'F08_si1922.mat', 'F08_si1915.mat', 'M02_si796.mat', 'F09_si2150.mat', 'F10_si2322.mat', 'F08_si1846.mat', 'F04_sx180.mat', 'F01_si614.mat', 'F03_si868.mat', 'F04_si1022.mat', 'M02_si674.mat', 'F07_si1599.mat', 'F02_si753.mat', 'F10_sx409.mat', 'M06_si1438.mat', 'F03_si831.mat', 'F03_si867.mat', 'F04_si1098.mat', 'F04_si1153.mat', 'F02_si664.mat', 'M06_si1408.mat', 'M06_si1429.mat', 'F06_si1421.mat', 'M02_si790.mat', 'M07_sx294.mat', 'F10_sx429.mat', 'F02_si769.mat', 'F04_sx161.mat', 'M02_si779.mat', 'F10_si2206.mat', 'M08_si1947.mat', 'M09_si2036.mat', 'M08_si1891.mat', 'F10_si2230.mat', 'M03_si989.mat', 'F02_si675.mat', 'F01_si610.mat', 'F05_sa2.mat', 'F05_sx185.mat', 'F10_si2286.mat', 'F02_si826.mat', 'M06_si1453.mat', 'M06_si1498.mat', 'M08_si1956.mat', 'F09_si2012.mat', 'M03_sx99.mat', 'F05_sx197.mat', 'F02_si716.mat', 'M07_si1641.mat', 'M07_sx273.mat', 'M07_sx282.mat', 'M06_si1445.mat', 'F04_si1085.mat', 'F04_si1059.mat', 'F08_si1842.mat', 'F10_si2241.mat', 'F05_si1356.mat', 'F10_si2179.mat', 'F09_si2093.mat', 'M07_si1691.mat', 'M08_si1959.mat', 'F04_si1172.mat', 'M04_si1040.mat', 'M02_si751.mat', 'M07_si1638.mat', 'F04_sx158.mat', 'F06_si1467.mat', 'F09_si2128.mat', 'F01_si641.mat', 'F01_si580.mat', 'F02_si714.mat', 'F10_sx451.mat', 'M07_sx317.mat', 'M03_sx128.mat', 'M03_si939.mat', 'F04_si1081.mat', 'F05_sx205.mat', 'F04_si1084.mat', 'F10_si2176.mat', 'M02_si758.mat', 'F08_si1888.mat', 'F01_si638.mat', 'F07_si1745.mat', 'F04_sx181.mat', 'M03_sx106.mat', 'M06_sx241.mat', 'M08_si1838.mat', 'F01_si616.mat', 'M07_si1743.mat', 'M07_si1607.mat', 'F08_si1963.mat', 'F02_si756.mat', 'F10_sx415.mat', 'M03_si956.mat', 'F02_si800.mat', 'F08_si1787.mat', 'F04_si1051.mat', 'F04_si1120.mat', 'M06_si1418.mat', 'F05_si1358.mat', 'M01_si634.mat', 'M02_si680.mat', 'F04_sx141.mat', 'M09_si2098.mat', 'F04_si1177.mat', 'F07_si1596.mat', 'F04_si1035.mat', 'M01_sx31.mat', 'M07_si1612.mat', 'M08_si1829.mat', 'F01_si604.mat', 'F10_si2267.mat', 'F10_si2283.mat', 'F02_si752.mat', 'M02_si792.mat', 'F01_si454.mat', 'F03_sx98.mat', 'M03_si875.mat', 'F09_si2139.mat', 'F10_sx450.mat', 'F04_si1094.mat', 'M07_si1753.mat', 'M03_si890.mat', 'M02_si753.mat', 'F08_si1957.mat', 'F02_si767.mat', 'F02_si747.mat', 'F07_si1603.mat', 'M07_si1738.mat', 'M03_si910.mat', 'M05_si1296.mat', 'F08_si1826.mat', 'M06_si1497.mat', 'M09_si1982.mat', 'F01_si464.mat', 'F01_si608.mat', 'F10_si2274.mat', 'M01_si510.mat', 'F02_si806.mat', 'F02_si808.mat', 'F02_si655.mat', 'F04_sx152.mat']
            constant = 1.0 # used for energy normalization

            # Female
            for spk_i in SPK_ID:
                spk_list = os.listdir(os.path.join(PTDB_path, spk_i))
                spk_list.sort()
                test_list = test_list + [PTDB_path + spk_i + '/' + s for s in spk_list if '.wav' in s]

            with torch.no_grad():

                dtr_avg_list = []
                vde_avg_list = []
                random.seed(1)


                start1 = timeit.default_timer()
                accu_test_dr = 0.0
                accu_test_vde = 0.0
                accu_test_rapt_vde = 0.0
                accu_test_nframes = 0
                ttime = 0.
                mtime = 0.
                cnt = 0.
                dtr_list = []

                for k, sndFile in enumerate(test_list):

                    (sr, audio) = wavfile.read(sndFile)
                    if sndFile.split('/')[-1][4:-4] + '.mat' in ptdb_tug_discarded_file:
                        print(k)
                        continue

                    if len(audio.shape) == 2:
                        audio = audio.mean(1)  # make mono
                    audio = audio.astype(np.float32)

                    if sr != model_srate: # resample audio if necessary
                        from resampy import resample
                        audio = resample(audio, sr, model_srate)

                    # energy normalization

                    audio = (audio - np.mean(audio))/np.std(audio)

                    rapt_est_f0 = pysptk.sptk.rapt(audio/max(abs(audio))*np.power(2,15), 8000, 80, min=60, max=404, voice_bias=0.0, otype='f0')
                    dio_est_f0, t = pw.dio(audio.astype(np.float64)/max(abs(audio.astype(np.float64)))*np.power(2,15), 8000, f0_floor=60.0, f0_ceil=404.0, frame_period=10)
                    # dio_est_f0, t = pw.dio(audio.astype(np.float64), 8000)

                    audio = np.squeeze(audio)
                    feat = self.stft(audio)
                    feat = np.swapaxes(feat, 1, 2)

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

                    # rapt_f0_voice = rapt_est_f0
                    # rapt_f0_voice[rapt_f0_voice>0]=1
                    # min_len = min([len(rapt_f0_voice), len(frequencies)])
                    # est_f0 = frequencies * rapt_f0_voice
                    est_f0 = frequencies * voicing_est

                    # Ref f0
                    ref_f0_path_arr = sndFile.split("/")
                    ref_f0_sentence = ref_f0_path_arr[-1].split('_')

                    ref_f0_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/PTDB-TUG/PTDB-TUG_pitch/ptdb_tug_consensus_' + ref_f0_sentence[1] + '_' + ref_f0_sentence[-1][:-4] + '/pitch.npy'
                    # ref_lar_f0_path = '/scratch/yixuanz/FDA_pitch/cstr_consensus_' + ref_f0_sentence[0][:-4] + '/lar_f0.npy'
                    ref_time_path = '/ocean/projects/cis210082p/yixuanz/monopitch/test/PTDB-TUG/PTDB-TUG_pitch/ptdb_tug_consensus_' + ref_f0_sentence[1] + '_' + ref_f0_sentence[-1][:-4] + '/time.npy'
                    ref_prob_path = '/ocean/projects/cis210082p/yixuanz/monopitch/test/PTDB-TUG/PTDB-TUG_pitch/ptdb_tug_consensus_' + ref_f0_sentence[1] + '_' + ref_f0_sentence[-1][:-4] + '/probability.npy'
                    
                    ref_voicing_path_arr = ref_f0_path_arr
                    ref_voicing_path_arr[10] = 'REF'
                    ref_voicing_path_arr[-1] = 'ref'+ref_voicing_path_arr[-1][3:-3]+'f0'
                    ref_voicing_path = '/'.join(ref_voicing_path_arr)

                    if os.path.exists(ref_f0_path):

                        ref_f0_mat = np.load(ref_f0_path)
                        ref_f0_ = ref_f0_mat
                        # ref_lar_f0 = np.load(ref_lar_f0_path)
                        # ref_time = np.load(ref_time_path)

                        with open(ref_voicing_path) as f:
                            ref_lines = f.readlines()
                            ref_voicing = [float(line.split(' ')[0]) for line in ref_lines]
                        ref_voicing = np.asarray(ref_voicing)
                        ref_voicing[ref_voicing>0] = 1
                        
                        min_len = min(len(ref_f0_), len(ref_voicing))

                        # ref_f0 = ref_voicing
                        ref_f0 = ref_f0_[:min_len] * ref_voicing[:min_len]

                        # print(len(ref_f0), len(est_f0))
                        ref_f0 = ref_f0[:]
                        est_f0 = est_f0[:]
                        min_len = min(len(ref_f0), len(est_f0))
                        ref_f0 = ref_f0[:min_len]
                        est_f0 = est_f0[:min_len]

                        if len(ref_f0[ref_f0!=0]) ==0 or min(ref_f0[ref_f0!=0]) < 30 or max(ref_f0) > 1000:
                            print("Out of bound!!")
                            continue

                        dtr = detection_rate(ref_f0, est_f0)
                        vde = voicing_decision_err(np.copy(ref_f0), np.copy(est_f0))
                        rapt_vde = voicing_decision_err(np.copy(ref_f0), np.copy(rapt_est_f0))

                        accu_test_dr += dtr
                        accu_test_vde += vde
                        accu_test_rapt_vde += rapt_vde
                        dtr_list.append(dtr)

                        cnt += 1
                        end = timeit.default_timer()
                        curr_time = end - start
                        ttime += curr_time
                        mtime = ttime / cnt
                        mtime = (mtime * (k) + (end-start)) / (k+1)
                        print('{}/{}, detection_rate = {:.4f}, voicing_decision_error = {:.4f}, time/utterance = {:.4f}, '
                               'mtime/utternace = {:.4f}'.format(k+1, len(test_list), dtr, vde, curr_time, mtime))
                        
                        # if k+1 == 100:
                        #     break
                        #     scipy.io.savemat('plot_ptdb.mat', {'mix':audio, 'ref_f0': ref_f0})
                        #     assert(1==2)

                avg_test_dr = accu_test_dr / cnt
                avg_test_vde = accu_test_vde / cnt
                avg_test_rapt_vde = accu_test_rapt_vde / cnt
                std = np.std(np.asarray(dtr_list))
                print('Average detection rate: ', avg_test_dr)
                print('Average voicing decision error: ', avg_test_vde)
                print('Average voicing decision error of RAPT: ', avg_test_rapt_vde)
                print("Standard deviation: ", std)
                # scipy.io.savemat('dr.mat', {'dtr_list': dtr_list})
                end1 = timeit.default_timer()
                print('********** Finisehe working on test files. time taken = {:.4f} **********'.format(end1 - start1))
                dtr_avg_list.append(avg_test_dr)
                vde_avg_list.append(avg_test_vde)
                assert(1==2)
        
            print(dtr_avg_list, vde_avg_list)       

    def test_keele(self, args, model_srate=8000, model_input_size=929):

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

            # test set
            PTDB_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/KEELE/'
            SPK_ID = ['f1nw0000', 'f2nw0000', 'f3nw0000', 'f4nw0000', 'f5nw0000', 'm1nw0000', 'm2nw0000', 'm3nw0000', 'm4nw0000', 'm5nw0000']
            test_list = []

            constant = 1.0 # used for energy normalization

            for spk_i in SPK_ID:
                spk_list = os.listdir(os.path.join(PTDB_path, spk_i))
                spk_list.sort()
                test_list = test_list + [PTDB_path + spk_i + '/' + s for s in spk_list if 'signal.wav' in s]

            with torch.no_grad():

                dtr_avg_list = []
                vde_avg_list = []
                random.seed(1)


                start1 = timeit.default_timer()
                accu_test_dr = 0.0
                accu_test_vde = 0.0
                accu_test_rapt_vde = 0.0
                accu_test_nframes = 0
                
                ttime = 0.
                mtime = 0.
                cnt = 0.
                dtr_list = []

                for k, sndFile in enumerate(test_list):
                    print(sndFile)
                    (sr, audio) = wavfile.read(sndFile)

                    if len(audio.shape) == 2:
                        audio = audio.mean(1)  # make mono
                    audio = audio.astype(np.float32)

                    if sr != model_srate: # resample audio if necessary
                        from resampy import resample
                        audio = resample(audio, sr, model_srate)

                    # energy normalization

                    audio = (audio - np.mean(audio))/np.std(audio)

                    rapt_est_f0 = pysptk.sptk.rapt(audio/max(abs(audio))*np.power(2,15), 8000, 80, min=60, max=404, voice_bias=0.0, otype='f0')
                    dio_est_f0, t = pw.dio(audio.astype(np.float64)/max(abs(audio.astype(np.float64)))*np.power(2,15), 8000, f0_floor=60.0, f0_ceil=404.0, frame_period=10)
                    # dio_est_f0, t = pw.dio(audio.astype(np.float64), 8000)

                    audio = np.squeeze(audio)
                    feat = self.stft(audio)
                    feat = np.swapaxes(feat, 1, 2)

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

                    # rapt_f0_voice = rapt_est_f0
                    # rapt_f0_voice[rapt_f0_voice>0]=1
                    # min_len = min([len(rapt_f0_voice), len(frequencies)])
                    # est_f0 = frequencies * rapt_f0_voice
                    est_f0 = frequencies * voicing_est

                    # Ref f0
                    ref_f0_path_arr = sndFile.split("/")
                    ref_f0_sentence = ref_f0_path_arr[-2].split('_')

                    ref_f0_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/KEELE/KEELE_pitch/keele_consensus_' + ref_f0_sentence[0] + '/pitch.npy'
                    # ref_lar_f0_path = '/scratch/yixuanz/FDA_pitch/cstr_consensus_' + ref_f0_sentence[0][:-4] + '/lar_f0.npy'
                    # ref_time_path = '/scratch/bbje/zhang23/data/monopitch/test/KEELE/KEELE_pitch/keele_consensus_' + ref_f0_sentence[1] + '_' + ref_f0_sentence[-1][:-4] + '/time.npy'
                    # ref_prob_path = '/scratch/bbje/zhang23/data/monopitch/test/KEELE/KEELE_pitch/keele_consensus_' + ref_f0_sentence[1] + '_' + ref_f0_sentence[-1][:-4] + '/probability.npy'
                    

                    ref_voicing_path = '/ocean/projects/cis210082p/yixuanz/data/monopitch/test/KEELE/' + ref_f0_sentence[0] + '/pitch.npy'

                    if os.path.exists(ref_f0_path):

                        ref_f0_mat = np.load(ref_f0_path)
                        ref_f0_ = ref_f0_mat
                        # ref_lar_f0 = np.load(ref_lar_f0_path)
                        # ref_time = np.load(ref_time_path)

                        # with open(ref_voicing_path) as f:
                        #     ref_lines = f.readlines()
                        #     ref_voicing = [float(line.split(' ')[0]) for line in ref_lines]
                        # ref_voicing = np.asarray(ref_voicing)
                        ref_voicing_mat = np.load(ref_voicing_path)
                        ref_voicing = [v[1] for v in ref_voicing_mat]
                        ref_voicing = np.asarray(ref_voicing)
                        # ref_voicing[ref_voicing<0] = 0
                        ref_voicing[ref_voicing>0] = 1

                        # print(len(ref_f0_), len(ref_voicing), len(est_f0))
                        # assert(1==2)

                        min_len = min(len(ref_f0_), len(ref_voicing))

                        ref_f0 = ref_voicing
                        # ref_f0 = ref_f0_[:min_len] * ref_voicing[:min_len]

                        # print(len(ref_f0), len(est_f0))
                        ref_f0=ref_f0[1:]
                        est_f0=est_f0[:]
                        min_len = min(len(ref_f0), len(est_f0))
                        ref_f0_ = ref_f0[:min_len].copy()
                        est_f0 = est_f0[:min_len]
                        
                        # remove all uncertain frames
                        # print(min(ref_f0))
                        # min_len = min(len(ref_f0_), len(rapt_est_f0))
                        rapt_est_f0 = rapt_est_f0[:min_len]
                        ref_f0_ = ref_f0[:min_len]
                        # ref_f0[ref_f0<0] = 2
                        est_f0 = np.asarray([f0 for i, f0 in enumerate(est_f0) if (ref_f0_[i]>=0)])
                        rapt_est_f0 = np.asarray([f0 for i, f0 in enumerate(rapt_est_f0) if (ref_f0_[i]>=0)])
                        ref_f0 = np.asarray([f0 for i, f0 in enumerate(ref_f0_) if (ref_f0_[i]>=0)])

                        # if min(ref_f0[ref_f0!=0]) < 30 or max(ref_f0) > 1000:
                        #     print("Out of bound!!")
                        #     continue

                        dtr = detection_rate(ref_f0, est_f0)
                        vde = voicing_decision_err(np.copy(ref_f0), np.copy(est_f0))
                        rapt_vde = voicing_decision_err(np.copy(ref_f0), np.copy(rapt_est_f0))

                        accu_test_dr += dtr
                        accu_test_vde += vde
                        accu_test_rapt_vde += rapt_vde
                        dtr_list.append(dtr)

                        cnt += 1
                        end = timeit.default_timer()
                        curr_time = end - start
                        ttime += curr_time
                        mtime = ttime / cnt
                        mtime = (mtime * (k) + (end-start)) / (k+1)
                        print('{}/{}, voicing_decision_error = {:.4f}, rapt_voicing_decision_error = {:.4f}, time/utterance = {:.4f}, '
                               'mtime/utternace = {:.4f}'.format(k+1, len(test_list), vde, rapt_vde, curr_time, mtime))
                        
                        # if k+1 == 2:
                        #     scipy.io.savemat('plot_keele_rapt.mat', {'mix':audio, 'ref_f0': ref_f0, 'est_f0': est_f0, 'rapt_est_f0': rapt_est_f0})
                        #     assert(1==2)
                
                    # scipy.io.savemat(self.output_dir + '/test'+str(k+1)+'.mat', {'est_pitch': frequencies, 'ref_f0': ref_f0})

                avg_test_dr = accu_test_dr / cnt
                avg_test_vde = accu_test_vde / cnt
                avg_test_rapt_vde = accu_test_rapt_vde / cnt
                std = np.std(np.asarray(dtr_list))
                print('Average detection rate: ', avg_test_dr)
                print('Average voicing decision error: ', avg_test_vde)
                print('Average voicing decision error of RAPT: ', avg_test_rapt_vde)
                print("Standard deviation: ", std)
                # scipy.io.savemat('dr.mat', {'dtr_list': dtr_list})
                end1 = timeit.default_timer()
                print('********** Finisehe working on test files. time taken = {:.4f} **********'.format(end1 - start1))
                dtr_avg_list.append(avg_test_dr)
                vde_avg_list.append(avg_test_vde)
                assert(1==2)        

            print(dtr_avg_list, vde_avg_list) 