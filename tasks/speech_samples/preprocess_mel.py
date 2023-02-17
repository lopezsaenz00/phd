#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Fri October 23 14:12:00 2020
From Mingjie Chen

@author: Jose Antonio Lopez @ The University of Sheffield

Creates Mel-Spectograms

"""

import os, sys, time, tempfile
sys.path.append('/aa/vae_ph/tools')
import load_txt

import h5py
import numpy as np
import sys,os
import glob
import librosa
import argparse
import json
import scipy
import scipy.io.wavfile as wav
from scipy.fftpack import dct

#import libraries for graphics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

'''
    phoneme label dict
'''


# PHONEME_DICT_61 = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5, 'ax-h': 6, 'axr': 7, 'ay': 8, 'b': 9, 'bcl': 10,
                # 'ch': 11, 'd': 12, 'dcl': 13, 'dh': 14, 'dx': 15, 'eh': 16, 'el': 17, 'em': 18, 'en': 19, 'eng': 20,
                # 'epi': 21, 'er': 22, 'ey': 23, 'f': 24, 'g': 25, 'gcl': 26, 'h#': 27, 'hh': 28, 'hv': 29, 'ih': 30,
                # 'ix': 31, 'iy': 32, 'jh': 33, 'k': 34, 'kcl': 35, 'l': 36, 'm': 37, 'n': 38, 'ng': 39, 'nx': 40,
                # 'ow': 41, 'oy': 42, 'p': 43, 'pau': 44, 'pcl': 45, 'q': 46, 'r': 47, 's': 48, 'sh': 49, 't': 50,
                # 'tcl': 51, 'th': 52, 'uh': 53, 'uw': 54, 'ux': 55, 'v': 56, 'w': 57,  'y': 58, 'z': 59, 'zh': 60}

# PHONEME_DICT_48 = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5, 'ax-h': 5, 'er': 6, 'axr': 6, 'ay': 7, 'b': 8,
                   # 'bcl': 9, 'dcl': 9, 'gcl': 9, 'ch': 10, 'd': 11, 'dh': 12, 'dx': 13, 'eh': 14, 'el': 15, 'em': 16,
                   # 'm': 16, 'en': 17, 'eng': 18, 'ng': 18, 'epi': 19, 'ey': 20, 'f': 21, 'g': 22, 'h#': 23, 'pau': 23,
                   # 'hh': 24, 'hv': 24, 'ih': 25, 'ix': 26, 'iy': 27, 'jh': 28, 'k': 29, 'kcl': 30, 'tcl': 30, 'pcl': 30,
                   # 'l': 31, 'n': 32, 'nx': 32, 'ow': 33, 'oy': 34, 'p': 35, 'r': 36, 's': 37, 'sh': 38, 't': 39,
                   # 'th': 40, 'uh': 41, 'uw': 42, 'ux': 42, 'v': 43, 'w': 44, 'y': 45, 'z': 46, 'zh': 47, 'q': 48}

PHONEME_DICT = { 'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5, 'ay': 6, 'b': 7, 'ch': 8, 'd': 9,
                    'dh': 10, 'ea': 11, 'eh': 12, 'el': 13, 'em': 14, 'en': 15, 'er': 16, 'ey': 17, 'f': 18, 'g': 19,
                    'hh': 20, 'ia': 21, 'ih': 22, 'iy': 23, 'jh': 24, 'k': 25, 'l': 26, 'm': 27, 'n': 28, 'ng': 29, 'oh': 30,
                    'ow': 31, 'oy': 32, 'p': 33, 'r': 34, 's': 35, 'sh': 36, 'silsp': 37, 't': 38, 'th': 39, 'ua': 40, 'uh': 41,
                    'uw': 42, 'v': 43, 'w': 44, 'y': 45, 'z': 46, 'zh': 47 }
                    

def extract_labels(labfile):
    '''This reads the lab file or GOP file and aligns each frame
    with the corresponding phone label'''

    lab = load_txt.load_txtfile_aslist(labfile)
    newlab = []
    labels = [] #the phone labels per frame
    time = [] #the timing information of each phone segment
    frames = [] #the number frame information of each phone segment
    scores = [] #the gop scores of equivalent
    ph_seq = [] #the phone sequence used for training the assessor's decision model
    
    skip_frames = True
    
    for line in lab:
        if len(line.split(' ')) > 3:
            elements = line.split(' ')
            start = float(elements[0])
            end = float(elements[1])
            nf = int(float(elements[1])*1e7/100000) - int(float(elements[0])*1e7/100000)
            ph = elements[2][elements[2].rfind('_')+1:]
            wd = elements[2][:elements[2].rfind('_')]
            
            labels = labels + ([PHONEME_DICT[ph]]*nf)
            
            if 'SENT_START' in wd and skip_frames:
                jump_frames = nf
                skip_frames = False
            
            if 'sil' not in ph:
                time.append([start, end])
                frames.append([int((start)*1e7/100000)-jump_frames, int((end)*1e7/100000)-jump_frames])
                scores.append(float(elements[-1]))
                ph_seq.append([ph, wd, str(PHONEME_DICT[ph]) ])
    
    
    return np.array(labels).astype(int), np.array(time).astype(float), np.array(frames).astype(int), np.array(scores).astype(float), ph_seq
    

def get_mfcc(y, args):
    '''
        get mfcc feature for one given filename based on librosa
        params:
            filenames: wav file path
            feature_config: configs for mfcc extraction based on librosa
        return: 
            wav
            mfcc 
        steps:
            1. librosa load in wav file
            2. librosa extract mfcc features

    '''
    #load in wav
    #y,sr = librosa.load(filename,sr = feature_config['sample_rate'])
    _mfcc = librosa.feature.mfcc(y = y, sr = args.sample_rate, n_fft=args.n_fft,
            hop_length=args.hop_length,   n_mels=80,n_mfcc=13)
    #[n_mfcc,T]
    _mfcc_delta = librosa.feature.delta(_mfcc)
    #[n_mfcc,T]
    _mfcc_delta2 = librosa.feature.delta(_mfcc,order=2)
    #[n_mfcc,T]
    mfcc = np.concatenate([_mfcc,_mfcc_delta,_mfcc_delta2],axis=0)
    #[3*n_mfcc,T]
    
    return mfcc # [3*n_mfcc,T]
 

def freq2mel(freq):
	"""Convert Frequency in Hertz to Mels

	Args:
		freq: A value in Hertz. This can also be a numpy array.

	Returns
		A value in Mels.
	"""
	return 2595 * np.log10(1 + freq / 700.0) 
    
    
def mel2freq(mel):
	"""Convert a value in Mels to Hertz

	Args:
		mel: A value in Mels. This can also be a numpy array.

	Returns
		A value in Hertz.
	"""
	return 700 * (10 ** (mel / 2595.0) - 1)
    
    
def get_melspectrogram(fname, args):
#this computes the mel spectogram (to dB) 

    name = fname.split('/')[-1]
    name = name[:name.rfind('.')]

    eps = 0.001					# Floor to avoid log(0)
    
    # load wav
    fs_hz, signal = wav.read(fname)
    signal_length = len(signal)

    nyquist = fs_hz / 2.0;		# Check the Nyquist frequency
    if args.high_freq_hz > nyquist:
        high_freq_hz = nyquist
    else:
        high_freq_hz =args.high_freq_hz

    # Pre-emphasis
    emphasised = np.append(signal[0], signal[1:] - args.preemph * signal[:-1]);

    # Compute number of frames and padding
    frame_length = int(round(args.frame_length_ms / 1000.0 * fs_hz));
    frame_step = int(round(args.frame_step_ms / 1000.0  * fs_hz));
    num_frames = int(np.ceil(float(signal_length - frame_length) / frame_step))
    print("number of frames is {}".format(num_frames))
    pad_signal_length = num_frames * frame_step + frame_length
    pad_zeros = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasised, pad_zeros)
    
    # Find the smallest power of 2 greater than frame_length
    NFFT = 1<<(frame_length-1).bit_length(); 
    
    # Compute mel-filters
    mel_filters = np.zeros((NFFT // 2 + 1, args.num_filters))
    low_freq_mel = freq2mel(args.low_freq_hz)
    high_freq_mel = freq2mel(high_freq_hz)
    mel_bins = np.linspace(low_freq_mel, high_freq_mel, args.num_filters + 2) # Equally spaced in the Mel scale
    freq_bins = mel2freq(mel_bins) # Convert Mel to Hz
    fft_bins = (NFFT + 1.0) * freq_bins // fs_hz # FFT bin indice for the filters
    for m in range(0, args.num_filters):
        for k in range(int(fft_bins[m]), int(fft_bins[m + 1])):
            mel_filters[k, m] = (k - fft_bins[m]) / (fft_bins[m+1] - fft_bins[m])
        for k in range(int(fft_bins[m + 1]), int(fft_bins[m + 2])):
            mel_filters[k, m] = (fft_bins[m + 2] - k) / (fft_bins[m + 2] - fft_bins[m+1])
            
    # Compute MFCCs

    # Hamming window
    win = np.hamming(frame_length)

    # Lifter
    lift = 1 + (args.cep_lifter / 2.0) * np.sin(np.pi * np.arange(args.num_ceps) / args.cep_lifter)

    # Pre-allocation
    feat_powspec = np.zeros((num_frames, NFFT//2+1)).astype(np.float32)
    feat_fbank = np.zeros((num_frames, args.num_filters)).astype(np.float32)
    feat_mfcc = np.zeros((num_frames, args.num_ceps)).astype(np.float32)

    # Compute MFCCs frame by frame
    for t in range(0, num_frames):

        # Framing
        frame = pad_signal[t*frame_step:t*frame_step+frame_length]

        # Apply the Hamming window
        frame = frame * win
        
        # Compute magnitude spectrum 
        magspec = np.absolute(np.fft.rfft(frame, NFFT))

        # Compute power spectrum
        powspec =  (magspec ** 2) * (1.0 / NFFT)

        # Save power spectrum features
        feat_powspec[t, :] = powspec;
        
        # Compute mel spectrum
        fbank = np.dot(powspec, mel_filters)
        
    # to decibel
    feat_fbank_dB = 20 * np.log10(np.maximum(1e-5,feat_fbank))
    feat_fbank_dB = np.clip(( feat_fbank_dB -100 + 20 ) / 100 , 1e-8, 1 )
    
    
    if name in args.test_sentences:
        # Log-compress power spectrogram
        feat_powspec[feat_powspec< eps] = eps
        feat_powspec = np.log(feat_powspec)
    
        # Plotting power spectrogram vs mel-spectrogram
        plt.figure(1)
        siglen = len(signal) / np.float(fs_hz);
        plt.subplot(311)
        plt.imshow(feat_powspec.T, origin='lower', aspect='auto', extent=(0,siglen,0,fs_hz/2000), cmap='gray_r')
        #plt.colorbar(c) 
        plt.title('Power Spectrogram\n'+name)
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().set_yticklabels(['',1,2,3,4,5,6,7,8])
        plt.ylabel('Frequency (kHz)')

        plt.subplot(312)
        freq_bins = freq_bins.astype(int)
        plt.imshow(feat_fbank.T, origin='lower', aspect='auto', extent=(0,siglen,0,args.num_filters), cmap='gray_r')
        if args.num_filters == 80:
            plt.yticks(np.arange(0,80,10))
            setlabels=['']
            for h in np.arange(0,80,10)[1:]:
                setlabels = setlabels + [round(freq_bins[h]/1000,2)]
            plt.gca().set_yticklabels(setlabels)
        else:
            plt.yticks([0,5,10,15,20,26])
            plt.gca().set_yticklabels(['',freq_bins[5],freq_bins[10],freq_bins[16],freq_bins[21],freq_bins[27]])
        plt.title('Mel-filter Spectrogram\n'+name)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (kHz)')
        
        plt.subplot(313)
        freq_bins = freq_bins.astype(int)
        plt.imshow(feat_fbank_dB.T, origin='lower', aspect='auto', extent=(0,siglen,0,args.num_filters), cmap='gray_r')
        if args.num_filters == 80:
            plt.yticks(np.arange(0,80,10))
            setlabels=['']
            for h in np.arange(0,80,10)[1:]:
                setlabels = setlabels + [round(freq_bins[h]/1000,2)]
            plt.gca().set_yticklabels(setlabels)
        else:
            plt.yticks([0,5,10,15,20,26])
        #plt.gca().set_yticklabels(['',freq_bins[5],freq_bins[10],freq_bins[16],freq_bins[21],freq_bins[27]])
        plt.title('Mel-filter Spectrogram (dB)\n'+name)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        
        plt.subplots_adjust(hspace = 1.2)
        
        plt.savefig('s/aa/vae_ph/task/melspec_'+str(args.num_filters)+'_'+name+'.png', dpi = 300)
    
    return feat_fbank_dB.astype(np.float32)
    

def read_phone(phone_path,wav_len):
    sentence_phone_labels = []
    pf = open(phone_path,'r+')
    for line in pf:
        #record phoneme start and end point and phoneme label
        sentence_phone_start = int(line.split(' ')[0])
        sentence_phone_end = int(line.split(' ')[1])
        # check boundary of phoneme
        if sentence_phone_start > wav_len or sentence_phone_end > wav_len :
            raise Exception
        # phoneme label
        phone = line.split(' ')[2].rstrip()
        # check if phoneme lable in phoneme 48 label collection
        if phone in PHONEME_DICT_61:
            phone_idx = PHONEME_DICT_61[phone]
        else:
            raise Exception
        sentence_phone_labels.append((sentence_phone_start, sentence_phone_end, phone_idx))
    pf.close()
    return sentence_phone_labels


def add_listofstring_data(h5py_file, h5py_filepath, data_list):
#from https://stackoverflow.com/questions/37873311/h5py-store-list-of-list-of-strings
#h5py cannot just have list of string data. It has to be converted into a (numpy) string object
#for this submit a list of lists of strings
#h5py_file : the h5py file to save
#h5py_filepath : the path for the data to save
#data_list : the list of lists of strings

    for i in range(len(data_list)):
        data_list[i] = np.array(data_list[i])
    
    data_list = np.vstack(data_list)
    data_list = data_list.astype(object)
    
    h5py_file.create_dataset(h5py_filepath,data=data_list, dtype=h5py.special_dtype(vlen=str))
    
    return h5py_file
    
def process(file_lists, lab_lists, dataset,speaker2index, f_h5, args ):

    max_scale = 0
    min_scale = 0
    
    data_length = []
    ph_length = []
    data_samples = []
    
    for fname in file_lists:
        print(f"{fname}",flush=True)

        directs = fname.split('/')

        speaker = directs[-1][directs[-1].rfind('-')+1:directs[-1].rfind('_')]
        
        #record speaker into global speaker2index
        if speaker not in speaker2index:
            speaker2index[speaker] = len(speaker2index)
        speaker_idx = speaker2index[speaker]
        wav_fname = directs[-1]
        

        mel = get_melspectrogram(fname, args)

        if np.max(mel) > max_scale:
            max_scale = np.max(mel)
        if np.min(mel) <min_scale:
            min_scale = np.min(mel)

        
        n_frames = mel.shape[0]
        #print(f"n_frames {n_frames}",flush=True)
        # get phone labels
        if len(fname.split('.')) <=1:
            raise Exception
            
            #the phone label array is for every frame. later we skip all the sil frames from the beggining and ending of the utterance
        phone_labels, time_info, frame_info, scores_info, ph_seq = extract_labels(lab_lists[directs[-1][:directs[-1].rfind('.')]])
        
        if len(phone_labels) != n_frames:
        
            if len(phone_labels) < n_frames:
                to_add = [37] * (n_frames - len(phone_labels))
                phone_labels = np.append(phone_labels, to_add)
            else:
                phone_labels = phone_labels[:n_frames]
                
        if len(phone_labels) != n_frames:
            raise Exception     
        
        #locate the start of the wav (the silsp at the beggining and the end)
        for counter, item in enumerate(phone_labels):
            if item != 37:
                break
                
        start = counter
        
        for counter, item in enumerate(reversed(phone_labels.tolist())):
            if item != 37:
                break  
        end = len(phone_labels)-counter
            
        mel = mel[start:end]
        y_val = phone_labels[start:end]
        reduced_length = mel.shape[0]
        
        if -1 in y_val:
            raise Exception 
            

        h5_path = dataset+'/'+ fname.split('.')[0].lower().split('/')[-1]
        print(f"h5_path {h5_path} start {start} end {end} n_frames {n_frames} reduced_length {reduced_length} mel {mel.shape} phone {y_val.shape} speaker {speaker} idx {speaker_idx}",flush=True)
        
        
        f_h5.create_dataset(f"{h5_path}/mel",data=mel,dtype=np.float32)
        f_h5.create_dataset(f"{h5_path}/speaker",data=speaker_idx,dtype=np.int32)
        f_h5.create_dataset(f"{h5_path}/phone",data=y_val,dtype=np.int32)
        f_h5.create_dataset(f"{h5_path}/time_info",data=time_info,dtype=np.float32)
        f_h5.create_dataset(f"{h5_path}/frame_info",data=frame_info,dtype=np.int32)
        f_h5.create_dataset(f"{h5_path}/scores_info",data=scores_info,dtype=np.float32)
        f_h5 = add_listofstring_data(f_h5, f"{h5_path}/ph_seq", ph_seq)
        
        data_length.append( (h5_path,reduced_length)  )
        
        #for every phone in ph_seq, it is a training sample
        for seg, ph_id in enumerate(ph_seq):
            frame_idx = frame_info[seg][:].tolist()
            ph_id_list = ph_id[:].tolist()
            data_samples.append([h5_path, seg, frame_idx[0] , frame_idx[1] ]+ph_id_list)
            ph_length.append(((h5_path,seg), frame_info[seg][:].tolist()[1] - frame_info[seg][:].tolist()[0]))
        
    return f_h5, speaker2index, data_length,max_scale,min_scale, data_samples, ph_length

    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main function
def main():

	'''
	args
	'''

	# all args have default
    

	parser = argparse.ArgumentParser()

	#parser.add_argument('--wav_path',type=str,default='timit/')
	parser.add_argument('--wav_scp_train',type=str,nargs=1, help='scp for wav train files',required=True)
	parser.add_argument('--wav_scp_test',type=str,nargs=1, help='scp for wav test files',required=True)
	parser.add_argument('--lab_scp_train',type=str,nargs=1, help='scp for lab train files',required=True)
	parser.add_argument('--lab_scp_test',type=str,nargs=1, help='scp for lab test files',required=True)
	parser.add_argument('--dataset_output', default="./h5/timit_mfcc39.h5")
	parser.add_argument('--ph_samp_output', default='./json/ph_samp_mfcc.json')
	parser.add_argument('--speaker2index_output',default='./json/speaker2index.json')
	parser.add_argument('--data_length_output',default='./json/data_length_mfcc.json')
	parser.add_argument('--ph_length_output',default='./json/ph_length_mfcc.json')
	parser.add_argument('--sample_rate',default=16000)
	parser.add_argument('--n_mfcc',default=13)
	parser.add_argument('--n_fft',default=int(0.025*16000))
	parser.add_argument('--hop_length',default=int(0.010*16000))
	parser.add_argument('--n_mels',default=80)
	parser.add_argument('--win_length',default=int(0.025*16000))
	parser.add_argument('--add_noise',default=False,action='store_true')
	parser.add_argument('--SNR',type=int,default=10)
    #for the melspec
	parser.add_argument('--frame_length_ms', help='frame length in ms',default=int(25))
	parser.add_argument('--frame_step_ms', help='frame shift in ms',default=int(10))
	parser.add_argument('--preemph', help='pre-emphasis factor',default=float(0.97))
	parser.add_argument('--low_freq_hz', help='low frequency in Hz',default=int(0))
	parser.add_argument('--high_freq_hz', help='High frequency in Hz',default=int(8000))
	parser.add_argument('--num_filters', help='number of mel-filters',default=int(26))
	parser.add_argument('--num_ceps', help='number of cepstral coefficients (excluding C0)',default=int(12))
	parser.add_argument('--cep_lifter', help='Cepstral liftering order',default=int(22))
    
	parser.add_argument('--test_sentences', help='Sentences used for debug',default=['sentence_1', 'sentence_2'])
	parser.add_argument('--f', default=False, action='store_true')

	args = parser.parse_args()
    
	args.num_filters = int(args.num_filters)

	print(args)
	# file lists
	
	train_lists = load_txt.load_txtfile_aslist(args.wav_scp_train[0])
	test_lists = load_txt.load_txtfile_aslist(args.wav_scp_test[0])
	
	_lablists_train = load_txt.load_txtfile_aslist(args.lab_scp_train[0])
	_lablists_test = load_txt.load_txtfile_aslist(args.lab_scp_test[0])

    
	labs_train = {}
	for lab in _lablists_train:
		name = lab[lab.rfind('/')+1: lab.rfind('_')]
		labs_train[name] = lab
	labs_test = {}
	for lab in _lablists_test:
		name = lab[lab.rfind('/')+1: lab.rfind('_')]
		labs_test[name] = lab
	
	
	print(f"train {len(train_lists)} test {len(test_lists)}",flush=True)
    
	DEBUG = args.f       # debug flag - dump temp folder in ITSLFATBASE
    
	if DEBUG:
		# locel folder if dumped 
		WID     = str(int(time.time()))
		WDIR    = os.path.join(BASE,WID)

		if not os.path.exists(WDIR):
			os.mkdir(WDIR)
	else:
		# system temp folder
		WDIR    = tempfile.mkdtemp();
		
		
	'''
	    create h5 file
	'''

	f_h5 = h5py.File(args.dataset_output,'w')


	'''
	    create global speaker dict
	'''
	speaker2index = {}	
	
	f_h5, speaker2index, train_data_length,tr_max,tr_min, tr_ph_samples, tr_ph_length = process(train_lists, labs_train, 'train', speaker2index, f_h5, args)
	f_h5, speaker2index, test_data_length,te_max,te_min, te_ph_samples, te_ph_length = process(test_lists, labs_test, 'test', speaker2index, f_h5, args)

    
	with open(args.speaker2index_output,'w') as sp_f:
	    json.dump(speaker2index,sp_f,indent=4)
	with open(args.data_length_output+'.train','w') as len_f_1:
	    json.dump(train_data_length,len_f_1,indent=4)
	with open(args.data_length_output+'.test','w') as len_f_2:
	    json.dump(test_data_length,len_f_2,indent=4)
	with open(args.ph_samp_output+'.train','w') as len_f_1:
	    json.dump(tr_ph_samples,len_f_1,indent=4)
	with open(args.ph_samp_output+'.test','w') as len_f_2:
	    json.dump(te_ph_samples,len_f_2,indent=4)	
	with open(args.ph_length_output+'.train','w') as len_f_1:
	    json.dump(tr_ph_length,len_f_1,indent=4)
	with open(args.ph_length_output+'.test','w') as len_f_2:
	    json.dump(te_ph_length,len_f_2,indent=4)
	    

	print(f"get train data instances {len(train_data_length)} ph seg {len(tr_ph_samples)} tr_max {tr_max} tr_min {tr_min}",flush=True)
	print(f"get test data instances {len(test_data_length)} ph seg {len(te_ph_samples)} te_max {te_max} te_min {te_min} ",flush=True)
    
    
if __name__ == '__main__':
    main()
