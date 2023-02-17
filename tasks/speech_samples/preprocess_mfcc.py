#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Fri October 23 14:12:00 2020
From Mingjie Chen

@author: Jose Antonio Lopez @ The University of Sheffield

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
import scipy.io.wavfile

'''
    phoneme label dict
'''


PHONEME_DICT_61 = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5, 'ax-h': 6, 'axr': 7, 'ay': 8, 'b': 9, 'bcl': 10,
                'ch': 11, 'd': 12, 'dcl': 13, 'dh': 14, 'dx': 15, 'eh': 16, 'el': 17, 'em': 18, 'en': 19, 'eng': 20,
                'epi': 21, 'er': 22, 'ey': 23, 'f': 24, 'g': 25, 'gcl': 26, 'h#': 27, 'hh': 28, 'hv': 29, 'ih': 30,
                'ix': 31, 'iy': 32, 'jh': 33, 'k': 34, 'kcl': 35, 'l': 36, 'm': 37, 'n': 38, 'ng': 39, 'nx': 40,
                'ow': 41, 'oy': 42, 'p': 43, 'pau': 44, 'pcl': 45, 'q': 46, 'r': 47, 's': 48, 'sh': 49, 't': 50,
                'tcl': 51, 'th': 52, 'uh': 53, 'uw': 54, 'ux': 55, 'v': 56, 'w': 57,  'y': 58, 'z': 59, 'zh': 60}

PHONEME_DICT_48 = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5, 'ax-h': 5, 'er': 6, 'axr': 6, 'ay': 7, 'b': 8,
                   'bcl': 9, 'dcl': 9, 'gcl': 9, 'ch': 10, 'd': 11, 'dh': 12, 'dx': 13, 'eh': 14, 'el': 15, 'em': 16,
                   'm': 16, 'en': 17, 'eng': 18, 'ng': 18, 'epi': 19, 'ey': 20, 'f': 21, 'g': 22, 'h#': 23, 'pau': 23,
                   'hh': 24, 'hv': 24, 'ih': 25, 'ix': 26, 'iy': 27, 'jh': 28, 'k': 29, 'kcl': 30, 'tcl': 30, 'pcl': 30,
                   'l': 31, 'n': 32, 'nx': 32, 'ow': 33, 'oy': 34, 'p': 35, 'r': 36, 's': 37, 'sh': 38, 't': 39,
                   'th': 40, 'uh': 41, 'uw': 42, 'ux': 42, 'v': 43, 'w': 44, 'y': 45, 'z': 46, 'zh': 47, 'q': 48}

PHONEME_DICT = { 'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5, 'ay': 6, 'b': 7, 'ch': 8, 'd': 9,
                    'dh': 10, 'ea': 11, 'eh': 12, 'el': 13, 'em': 14, 'en': 15, 'er': 16, 'ey': 17, 'f': 18, 'g': 19,
                    'hh': 20, 'ia': 21, 'ih': 22, 'iy': 23, 'jh': 24, 'k': 25, 'l': 26, 'm': 27, 'n': 28, 'ng': 29, 'oh': 30,
                    'ow': 31, 'oy': 32, 'p': 33, 'r': 34, 's': 35, 'sh': 36, 'silsp': 37, 't': 38, 'th': 39, 'ua': 40, 'uh': 41,
                    'uw': 42, 'v': 43, 'w': 44, 'y': 45, 'z': 46, 'zh': 47 }
                    

def extract_phone_labels(labfile):
    '''This reads the lab file or GOP file and aligns each frame
    with the corresponding phone label'''

    lab = load_txt.load_txtfile_aslist(labfile)
    newlab = []
    labels = []
    
    for line in lab:
        if len(line.split(' ')) > 3:
            elements = line.split(' ')
            
            nf = int(float(elements[1])*1e7/100000) - int(float(elements[0])*1e7/100000)
            ph = elements[2][elements[2].rfind('_')+1:]
            
            labels = labels + ([PHONEME_DICT[ph]]*nf)
    
    
    return np.array(labels).astype(int)
    

def get_mfcc(fname, args):
#this computes the mfcc

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
    feat_powspec = np.zeros((num_frames, NFFT//2+1))
    feat_fbank = np.zeros((num_frames, args.num_filters))
    feat_mfcc = np.zeros((num_frames, args.num_ceps))

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

        # # Compute log mel spectrum (cepstrum) - to obtain the MFCC
        fbank = np.dot(powspec, mel_filters)
        fbank[fbank < eps] = eps # Avoid log(0)
        fbank = np.log(fbank)

        # Save fbank features
        feat_fbank[t, :] = fbank

        # Apply DCT to get num_ceps MFCCs, omit C0
        mfcc = dct(fbank, norm='ortho')[1:num_ceps+1] # Omit C0

        # Liftering
        mfcc *= lift 

        # Save mfcc features
        feat_mfcc[t, :] = mfcc
        
        
    # Cepstral mean and variance normalisation
    feat_mfcc_z = (feat_mfcc - np.mean(feat_mfcc, axis=0)) / np.std(feat_mfcc, axis=0)
    
    
    # Log-compress power spectrogram
    feat_powspec[feat_powspec< eps] = eps
    feat_powspec = np.log(feat_powspec)
    
    
    
    if name in args.test_sentences:
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
        
        plt.savefig('/aa/vae_ph/task/melspec_'+str(args.num_filters)+'_'+name+'.png')
    
    return feat_fbank, feat_fbank_dB

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



def process(file_lists, lab_lists, dataset,speaker2index, f_h5, args ):

    
    max_scale = 0
    min_scale = 0
    
    data_length = []
    
    for fname in file_lists:
        print(f"{fname}",flush=True)

        directs = fname.split('/')

        speaker = directs[-1][directs[-1].rfind('-')+1:directs[-1].rfind('_')]
        
        #record speaker into global speaker2index
        if speaker not in speaker2index:
            speaker2index[speaker] = len(speaker2index)
        speaker_idx = speaker2index[speaker]
        wav_fname = directs[-1]
        

        mel, meldB = get_melspectrogram(fname, args)

        if np.max(mel) > max_scale:
            max_scale = np.max(mel)
        if np.min(mel) <min_scale:
            min_scale = np.min(mel)

        
        n_frames = mel.shape[0]
        #print(f"n_frames {n_frames}",flush=True)
        # get phone labels
        if len(fname.split('.')) <=1:
            raise Exception
            
        phone_labels = extract_phone_labels(lab_lists[directs[-1][:directs[-1].rfind('.')]])
        
        if len(phone_labels) != n_frames:
            phone_labels = phone_labels.tolist()
            while len(phone_labels) != n_frames:
                phone_labels.append(37)
                
            phone_labels = np.array(phone_labels)
        
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
        meldB = meldB[start:end]
        y_val = phone_labels[start:end]
        reduced_length = mel.shape[0]
        
        if -1 in y_val:
            raise Exception

        h5_path = dataset+'/'+ fname.split('.')[0].lower().split('/')[-1]
        print(f"h5_path {h5_path} start {start} end {end} n_frames {n_frames} reduced_length {reduced_length} mel {mel.shape} meldB {meldB.shape} phone {y_val.shape} speaker {speaker} idx {speaker_idx}",flush=True)

        
        
        f_h5.create_dataset(f"{h5_path}/mel",data=mel,dtype=np.float32)
        f_h5.create_dataset(f"{h5_path}/meldB",data=meldB,dtype=np.float32)
        f_h5.create_dataset(f"{h5_path}/speaker",data=speaker_idx,dtype=np.int32)
        f_h5.create_dataset(f"{h5_path}/phone",data=y_val,dtype=np.int32)
        data_length.append( (h5_path,reduced_length)  )
        
        
    return f_h5, speaker2index, data_length,max_scale,min_scale


    
    
    
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
	parser.add_argument('--speaker2index_output',default='./json/speaker2index.json')
	parser.add_argument('--data_length_output',default='./json/data_length_mfcc.json')
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
    
	parser.add_argument('--test_sentences', help='Sentences used for debug',default=['sentence_1', 'sentence_1'])
	parser.add_argument('--f', default=False, action='store_true')

	args = parser.parse_args()
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
	
	print("Example: "+train_lists[0])
	
	f_h5, speaker2index, train_data_length,tr_max,tr_min = process(train_lists, labs_train, 'train', speaker2index, f_h5, args)
	print(f"get train data instances {len(train_data_length)} tr_max {tr_max} tr_min {tr_min}",flush=True)
    
	f_h5, speaker2index, test_data_length,te_max,te_min = process(test_lists, labs_test, 'test', speaker2index, f_h5, args)
	print(f"get test data instances {len(test_data_length)} te_max {te_max} te_min {te_min} ",flush=True)
    
	with open(args.speaker2index_output,'w') as sp_f:
	    json.dump(speaker2index,sp_f,indent=4)
	with open(args.data_length_output+'.train','w') as len_f_1:
	    json.dump(train_data_length,len_f_1,indent=4)
	with open(args.data_length_output+'.test','w') as len_f_2:
	    json.dump(test_data_length,len_f_2,indent=4)	
	
    
if __name__ == '__main__':
    main()
