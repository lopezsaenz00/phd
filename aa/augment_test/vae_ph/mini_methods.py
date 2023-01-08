#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sat February 13 17:28:00 2021
From Mingjie Chen/Thomas Hain

@author: Jose Antonio Lopez @ Infirmary Rd

"""

import scipy.io.wavfile as wav
import numpy as np


#import libraries for graphics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors





def get_melspectrogram(fname, args ):
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
    # Here you can choose either the off-line mode, i.e. save all the frames in a
    # matrix and process them in one go, or the online mode, i.e. compute MFCCs 
    # frame by frame.

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

        # # Compute log mel spectrum (cepstrum) - to obtain the MFCC
        # fbank = np.dot(powspec, mel_filters)
        # fbank[fbank < eps] = eps # Avoid log(0)
        # fbank = np.log(fbank)

        # Save fbank features
        feat_fbank[t, :] = fbank

        # # Apply DCT to get num_ceps MFCCs, omit C0
        # mfcc = dct(fbank, norm='ortho')[1:num_ceps+1] # Omit C0

        # # Liftering
        # mfcc *= lift 

        # # Save mfcc features
        # feat_mfcc[t, :] = mfcc
        
    # to decibel
    feat_fbank_dB = 20 * np.log10(np.maximum(1e-5,feat_fbank))
    feat_fbank_dB = np.clip(( feat_fbank_dB -100 + 20 ) / 100 , 1e-8, 1 )
    
    siglen = len(signal) / np.float(fs_hz)
    
    return feat_fbank,  feat_fbank_dB.astype(np.float32), freq_bins, siglen
    
    
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
    
    
def plot_mel(mel, title, name, freq_bins, siglen, args):

    #plt.figure(1,1,1)
    fig, ax = plt.subplots()
    freq_bins = freq_bins.astype(int)
    ax.imshow(mel.T, origin='lower', aspect='auto', extent=(0,siglen,0,80), cmap='gray_r')
    if args.num_filters == 80:
        plt.yticks(np.arange(0,80,10))
        setlabels=['']
        for h in np.arange(0,80,10)[1:]:
            setlabels = setlabels + [round(freq_bins[h]/1000,2)]
        plt.gca().set_yticklabels(setlabels)
    else:
        plt.yticks([0,5,10,15,20,26])
    #plt.gca().set_yticklabels(['',freq_bins[5],freq_bins[10],freq_bins[16],freq_bins[21],freq_bins[27]])
    #plt.title('Mel-filter Spectrogram (dB)\n'+name)
    plt.title(f"{title}\n{name}")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    plt.subplots_adjust(hspace = 1.2)
    savename = f"/share/mini1/res/t/asr/call/childread-nl/its/aa/augment_test/augment_demo/{title}_{name}.png"
    plt.savefig(savename, dpi = 300)
    print(f"plot saved as:{savename}")

