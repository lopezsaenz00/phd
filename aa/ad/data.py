import torch.utils.data as data
import pandas as pd
import json
import pickle
import torch
import numpy as np
import h5py
import os
from models.vae_model import Model # VAE
import sys
sys.path.append('/aa/ad/tools')
from auxfunc import *

####select if cuda is available
#use_cuda = torch.cuda.is_available()
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

device = torch.device("cuda:"+str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")


#gender_f = open('json/speaker2gender.json','r')
#speaker2gender = json.load(gender_f)
class INADataset(data.Dataset):

    def __init__(self, samp_json_pa='./json/sample.json.train', h5_pa='./h5/timit.h5', ph_len='./json/data_length_plptrain.json' , with_speaker = False,with_phone=False, mvn_pa='./mvn/wav_mel80_train', vae_name = './vae_INA.pkl',vae_in_dim = 80, vae_hid = 32, seg_len = 20, lda_post = './INA.lda.csv.gz', lda_t = 4, lda_d = 256, lda_e = 20, use_cuda = False, ref_file = './ina.ref', ref_mode = 'and', gop = False, vae = False, lda = False,  phseg = False, phone = False, cntxt = False, word = False, preref = False, filt = None ):
    
        #get input flags
        #if true, include the inputs
        self.gop = gop
        self.vae = vae
        self.lda = lda
        self.filt = filt
        
        self.h5_file = h5py.File(h5_pa,'r')
        samp_f = open(samp_json_pa,'r')
        self.samples = json.load(samp_f)
        self.seg_len = seg_len
        self.ph_len_pa = ph_len #data_len before
        
        self.with_phone = phone
        self.with_word = word
        self.phseg = phseg
        self.with_cntxt = cntxt
        self.preref = preref
        
        print(f"Complete dataset samples: {len(self.samples)}")
        #check if the data has to be filtered
        if not(self.filt is None):
            #load the file and drop out the rejected segments
            filtdf = pd.read_csv(self.filt, sep = ' ', header=None)
            filtdf.columns = ['name', 'phidx', 'st', 'end', 'ph', 'wd', 'phtag','target', 'pos']
            filtdf = filtdf.drop(['target','pos'], axis = 1) 
            filtdf = filtdf.values.tolist()
            filtdf = [ '_'.join([ str(j) for j in sentence ]) for sentence in filtdf ]
            #at this point the list should be similar format of the segment list
            samplestring = [ '_'.join([ str(j) for j in sentence ]) for sentence in  self.samples ]
            
            #check which samples are contained in the filtered data
            self.tokeep = []
            for i, samp in enumerate (samplestring):
                if samp in filtdf:
                    self.tokeep.append(i)
            
            #keep the segments authorized
            self.samples = [self.samples[i] for i in self.tokeep]
            print(f"Resulted samples after filtering: {len(self.samples)}")

            
            
        
        #for the VAE
        if self.vae:
            self.vae_name = vae_name
            self.vae_in_dim = vae_in_dim
            self.vae_hid = vae_hid
            self.seg_len = seg_len
            self.vae_model = self.load_vae_model()
            self.use_cuda = use_cuda
            if self.use_cuda:
                self.vae_model.cuda()
        
        #for lda topic posteriors
        #lda list is an index for the utterance name
        #lda_pz is a matrix shape (utterance x topic posterior)
        if self.lda:
            self.lda_list, self.lda_pz = self.load_lda_pz(lda_post)
            self.lda_t = lda_t
            self.lda_d = lda_d
            self.lda_e = lda_e
        
        #annotation reference file
        self.ref_file = ref_file
        self.ref_mode = ref_mode
        #generate reference
        self.y, self.phref, self.phclass, self.leftphref, self.rightphref, self.wdref, self.prey, self.phsegref = self.gen_ref( self.ref_file, self.ref_mode )
        
        #get classes counts and loss-weights
        self.pos_weight = self.y.count(0)*1.0 / self.y.count(1)*1.0
        
        
        ##obtain mel_mvn, GOP_mvn, lda_mvn vectors
        if not os.path.exists(mvn_pa):
            os.makedirs(mvn_pa)
            
        if self.vae: 
            if not(self.filt is None):
                mvn_wav_name  = f"wav_mel80_{os.path.basename(self.filt)}.pkl"
            else:    
                mvn_wav_name  = 'wav_mel80.pkl'
             
            if not os.path.exists(os.path.join(mvn_pa,mvn_wav_name)):
                self.mel_mvn = self.compute_mel_mvn()
                #print(f"get mel mvn {self.mel_mvn}")
                print(f"get mel mvn.")
                with open(os.path.join(mvn_pa, mvn_wav_name),"wb") as mvn_f:
                    pickle.dump(self.mel_mvn, mvn_f)
            else:
                mvn_f = open(os.path.join(mvn_pa, mvn_wav_name),'rb')
                self.mel_mvn = pickle.load(mvn_f)
                #print(f"load mel mvn {self.mel_mvn}")  
                print(f"loaded mel mvn.")  
                
            if not(self.filt is None):
                emb_mvn_name  =  f"{self.vae_hid}emb_mvn_{os.path.basename(self.filt)}.pkl"
            else:    
                emb_mvn_name  = f"{self.vae_hid}emb_mvn.pkl"
                
            if not os.path.exists(os.path.join(mvn_pa, emb_mvn_name)):
                self.emb_mvn = self.compute_emb_mvn() 
                print(f"get emb mvn.")
                with open(os.path.join(mvn_pa, emb_mvn_name),"wb") as mvn_f:
                    pickle.dump(self.emb_mvn, mvn_f)
            else:
                mvn_f = open(os.path.join(mvn_pa, emb_mvn_name),'rb')
                self.emb_mvn = pickle.load(mvn_f)
                #print(f"load GOP mvn {self.lda_mvn}")
                print(f"load emb mvn.")  
        

        if self.gop:    
        
            if not(self.filt is None):
                gop_mvn_name  = f"INA_GOP_{os.path.basename(self.filt)}.pkl"
            else:    
                gop_mvn_name  = 'INA_GOP.pkl'
        
            if not os.path.exists(os.path.join(mvn_pa, gop_mvn_name)): 
                self.gop_mvn = self.compute_gop_mvn()
                #print(f"get GOP mvn {self.gop_mvn}")
                print(f"get GOP mvn.")
                with open(os.path.join(mvn_pa, gop_mvn_name),"wb") as mvn_f:
                    pickle.dump(self.gop_mvn, mvn_f)
            else:
                mvn_f = open(os.path.join(mvn_pa, gop_mvn_name),'rb')
                self.gop_mvn = pickle.load(mvn_f)
                #print(f"load GOP mvn {self.gop_mvn}")
                print(f"loaded GOP mvn.")
                
                
        if self.lda:         
        
            if not(self.filt is None):
                lda_mvn_name  = f"INA_LDA_t{self.lda_t}_d{self.lda_d}_e{self.lda_e}_{os.path.basename(self.filt)}.pkl"
            else:    
                lda_mvn_name  = f"INA_LDA_t{self.lda_t}_d{self.lda_d}_e{self.lda_e}.pkl"
        
            if not os.path.exists(os.path.join(mvn_pa, lda_mvn_name)):    
                self.lda_mvn = self.compute_lda_mvn()
                #print(f"get LDA mvn {self.lda_mvn}")
                print(f"get LDA mvn.")
                with open(os.path.join(mvn_pa,lda_mvn_name),"wb") as mvn_f:
                    pickle.dump(self.lda_mvn, mvn_f)
            else:
                mvn_f = open(os.path.join(mvn_pa, lda_mvn_name),'rb')
                self.lda_mvn = pickle.load(mvn_f)
                #print(f"load GOP mvn {self.lda_mvn}")
                print(f"loaded LDA mvn.")
            
            
        #to avoid using the vae model at every batch, 
        #we just generate the features we need.  
        if self.vae: 
            self.emb = self.obtain_norm_emb()
            
        #get the dimension of the features,
        #include phone tag, gop score, embedding and LDA
        input_dim = 1 #the phone tag
        if self.gop:
            input_dim += 1
        if self.vae:
            input_dim += len(self.emb_mvn['mean'])
        if self.lda:
            input_dim += len(self.lda_mvn['mean'])
        
        self.input_dim = input_dim
        
        
    def gen_ref(self, REF_FILE, REF_MODE):
    #this method reads the ref file.
        #REF_FILE = self.ref_file
        #REF_MODE = self.ref_mode
        ref = {}
        dref = []
        phref = []
        wdref = []
        phsegref = []

        with open(REF_FILE, 'r') as myfile:
            lines = myfile.read().splitlines()
            
            for line in lines:
            
                if REF_MODE == 'a1' or REF_MODE == 'a2' or REF_MODE == 'a3' :
                    line2append = line.split('\t')
                else:
                    line2append = line.split(' ')
                init = line2append[0]
                
                if 'ina-' in init:
                
                    filename = init
                    seg_score = []  
                    
                if len(line2append) == 4 and not( '---' in init):
                    seg_score.append(line2append)
                    
                if init == '.':
                    ref[filename] = seg_score
                    
                    
        # collect the annotators decision
        leftph = []
        rightph = []
        preref = []
        oldwd = ''
        for i, samp in enumerate(self.samples):
            filename = os.path.basename(samp[0])
            file_idx = samp[1]
            ph = samp[4]
            wd = samp[5]
            phid = samp[6]
            
            if file_idx == 0:
                leftph.append( 'start' )
                preref.append(-1)
            else:
                leftph.append( self.samples[i-1][4] )
                preref.append( 1 - int(ref[filename][file_idx-1][-1]))# append the decision of the previous segment of the same recording
            #print(f"i: {i}")   
            if i < len(self.samples)-1:
                if self.samples[i+1][1] == 0:
                    rightph.append( 'end' )
                else:    
                    rightph.append( self.samples[i+1][4] )
            else:
                rightph.append( 'end' )
            
            
            #the ref file
            ref_seg = ref[filename][file_idx]
            
            if not( wd == oldwd ) :
                oldwd = wd #update old word
                segcount = 0 #restar the segment counter. This is dependant on the word.
            
            dref.append( 1 - int(ref_seg[-1]))
            phref.append(ph)
            wdref.append(wd)
            phsegref.append(segcount)    
            segcount += 1
            
            
        #generate labels for phone symbols
        copphref = list(set(phref+['start', 'end']))
        phclass = {} 
        for num, ph in enumerate(copphref):
            phclass[ph] = num*1.0
        #generate labels for word symbols
        copphref = list(set(wdref))
        wdclass = {} 
        for num, wd in enumerate(copphref):
            wdclass[wd] = num*1.0        
            
            
        phref = np.array([ phclass[ph] for ph in phref] )
        phref = (phref - phref.mean() ) / phref.std()
        
        leftphref = np.array([ phclass[ph] for ph in leftph] )
        leftphref = (leftphref - leftphref.mean() ) / leftphref.std()
        
        rightphref = np.array([ phclass[ph] for ph in rightph] )
        rightphref = (rightphref - rightphref.mean() ) / rightphref.std()
        
        wdref = np.array([ wdclass[wd] for wd in wdref] )
        wdref = (wdref - wdref.mean() ) / wdref.std()
        
        preref = np.array(preref)
        preref = (preref - preref.mean() ) / preref.std()
        
        phsegref = np.array(phsegref)
        phsegref = (phsegref - phsegref.mean() ) / phsegref.std()

        return dref, phref, phclass, leftphref, rightphref, wdref, preref, phsegref
        
            
    def load_lda_pz(self,lda_post):
    	#loads the posterior probabilities and creates a small dictionary
    	#for indexing the probabilities given the utterance
    	
        lda_file = pd.read_csv(lda_post, index_col=0, sep=',', compression='gzip')
    	
        ##get data and labels
        topics=[name[name.rfind('z')+1:] for name in lda_file.columns if 'pz' in name and 'max' not in name]	
        pz_data = lda_file[ ['pz'+str(i) for i in topics] ].to_numpy()
        sentence = [name[name.rfind('.ina-')+1:name.rfind('.vq')] for name in lda_file['doc']]
        
        #if a recording doesnt appear on the filtered data, take it out.
        if not(self.filt is None):
            #sentence example ina-ndpt51516900hmdl_ndpt5123f85d97
            #self.sample example train/ina-ndpt51645663hmdl_ndpt5107f94d96 23 246 261 iy INCREASES 23
            filt_sentece = [ i[0][i[0].rfind('/')+1:] for i in self.samples ] #change the format of the self.sample to the lda pz
            filt_sentece = list( set(filt_sentece))
            
            tokeep = []
            for i, j in enumerate(sentence):
                if j in filt_sentece:
                    tokeep.append(i)
                    
            sentence = [sentence[i] for i in tokeep]
	
        return sentence, pz_data
    	
            
    def load_vae_model(self,):
    	#load the pre-trained vae model
        model = eval('Model')(c_in=self.vae_in_dim,hid=self.vae_hid,seg_len=self.seg_len)
        load_model_dir = self.vae_name
        print("This is lad_model_dir")
        print(load_model_dir)
        if not os.path.exists(load_model_dir):
            raise Exception
        model_f = open(load_model_dir,'rb')
        state_dict = torch.load(model_f)
        model.load_state_dict(state_dict['model'])
        model.eval()
        print("VAE model loaded.\n")
        
        return model
        
        
    def get_vae_emb(self, samp):
    #it generates the encoding for the phone segment
    	#input the self.samples items
    	h5_path = samp[0]
    	start = samp[2]
    	end = samp[3]
    	#original melspect
    	mel = self.h5_file[f'{h5_path}/mel'][()]
    	length = end - start
    	utt_length = mel.shape[0]
    	
    	while length < self.seg_len:
            end += 1
            length = end - start
            if length == self.seg_len:
            	break
            start -= 1
            length = end - start
            if length == self.seg_len:
            	break
        #for when the phone segment is waay too long	
    	while length > self.seg_len:
            end -= 1
            length = end - start
            if length == self.seg_len:
            	break
            start += 1
            length = end - start
            if length == self.seg_len:
            	break
    	
    	#this is for zero padding
    	if start < 0 or end > utt_length:
            pad = np.zeros((self.seg_len, self.vae_in_dim ))
            pad = pad.astype(np.float32)
            mel = self.h5_file[f'{h5_path}/mel'][()][max(start,0):min(end,utt_length) ]
            
            if start < 0:
                pad[abs(start):mel.shape[0]+abs(start),:mel.shape[1]] = mel
            if end > utt_length:
                pad[:mel.shape[0],:mel.shape[1]] = mel
            
            mel = pad
    	else:
            mel = self.h5_file[f'{h5_path}/mel'][()][start:end]
            
    	mel = (mel - self.mel_mvn['mean']) / self.mel_mvn['std']
    	mel = torch.from_numpy(mel).unsqueeze(0)
    	mel = mel.permute(0,2,1)
    	if self.use_cuda:
            mel = mel.cuda()
    	emb = self.vae_model.encode(mel)	
    	
    	return emb.detach()
      
      
    def compute_emb_mvn(self, ):
    #compute the mvn for the embedding
        x,x2,n = 0.,0.,0.
        
        for samp in self.samples:
            #h5_path = samp[0]
            #segment_id = samp[1]
            #start = samp[2]
            #end = samp[3]
            #ph = samp[4]
            #wd = samp[5]
            #phid = samp[6]
        
            emb = self.get_vae_emb(samp)
            emb = torch.flatten(emb)
            emb = emb.cpu().data.numpy()
            
            x += emb
            ##
            #here we wont use log add for multiplication because 
            #numbers can be negative
            x2 += emb **2
            n += 1
        mean = x/n
        std = np.sqrt(x2/n - mean**2)
        
        #this is the overall mean and std vector
        #mean = np.concatenate( [ np.array([self.phref.mean(), self.gop_mvn['mean']]), mean, self.lda_mvn['mean'] ] )
        #std = np.concatenate( [ np.array([self.phref.std(), self.gop_mvn['std']]), std, self.lda_mvn['std'] ] )
        
        return {'mean':mean,'std':std}
        
    def obtain_norm_emb(self,):
        #this method obtains the phone embedding from
        #the vae and stores the normalized vectors
        embeddings = np.empty((len(self.samples), len(self.emb_mvn['mean'])))
        
        for i, samp in enumerate(self.samples):
            #h5_path = samp[0]
            #segment_id = samp[1]
            #start = samp[2]
            #end = samp[3]
            #ph = samp[4]
            #wd = samp[5]
            #phid = samp[6]
        
            emb = self.get_vae_emb(samp)
            emb = torch.flatten(emb)
            emb = emb.cpu().data.numpy()
            
            #save the normalise 
            emb = (emb - self.emb_mvn['mean']) / self.emb_mvn['std']
            
            embeddings[i, :] = emb
            
        
        return embeddings
            
            
    def compute_mel_mvn(self,):
        if not os.path.exists(self.ph_len_pa):
            raise Exception
        data_len_f = open(self.ph_len_pa)
        data_lens = json.load(data_len_f)
        
        #check if the data is filtered (posterior)
        if not(self.filt is None):
            data_lens = [data_lens[i] for i in self.tokeep]
        
        x,x2,n = 0.,0.,0.
        #due to format mismatch. extract the names of the 
        #utterances and get the list without repetitions
        data_list = []
        for h5_path,sam_len in data_lens:
            data_list.append(h5_path[0])
            
        data_len_f.close()
        data_lens = list(set(data_list))

        for h5_path in data_lens:
            feat = self.h5_file[f'{h5_path}/mel'][()]
            x += np.sum(feat,axis=0,keepdims=True)
            x2 += np.sum(feat **2,axis=0,keepdims=True)
            n += feat.shape[0]
        mean = x/n
        std = np.sqrt(x2/n - mean**2)
        return {'mean':mean,'std':std}
        
    def compute_gop_mvn(self,):
        if not os.path.exists(self.ph_len_pa):
            raise Exception
        data_len_f = open(self.ph_len_pa)
        data_lens = json.load(data_len_f)
        
        #check if the data is filtered (posterior)
        if not(self.filt is None):
            data_lens = [data_lens[i] for i in self.tokeep]
        
        x,x2,n = 0.,0.,0.

        for h5_path, seglen in data_lens:
            #h5_path[0] utterance
            #h5_path[1] number of segment
            feat = self.h5_file[f'{h5_path[0]}/scores_info'][()][h5_path[1]]
            x += feat
            x2 += feat **2
            n += 1
        mean = x/n
        std = np.sqrt(x2/n - mean**2)
        data_len_f.close()
        return {'mean':mean,'std':std}
        
    def compute_lda_mvn(self,):
        #compute the mean and variance of the LDA posteriors
        #lucky it's already on a 2d numpy array
        #if using a filtered set, the 2d array has already been filtered out when the dataset was loaded.
        #mean = np.mean(self.lda_pz, 0)
        #std = np.std(self.lda_pz, 0)
        
        ##########################
        lda_pz = [ i[0][i[0].rfind('/')+1:] for i in self.samples ] #change the format of the self.sample to the lda pz           
        lda_pz = [ self.lda_pz[self.lda_list.index(i),:] for i in lda_pz]
        lda_pz = np.stack(lda_pz, axis=0)  #generate an array with the actual proportion of lda posterior observations used in the subset
        
        mean = np.mean(lda_pz, 0)
        std = np.std(lda_pz, 0)    
        
        return {'mean':mean,'std':std}


    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self,i):
        samp = self.samples[i]
        h5_path = samp[0] #example train/ina-ndpt51493796hmdl_ndpt5107f94d96
        segment_id = samp[1]
        start = samp[2]
        end = samp[3]
        ph = samp[4]
        wd = samp[5]
        phid = samp[6]
        
        speaker_name = os.path.basename(h5_path).split('_')[-1]
        
        #obtain the assessor decision
        #y = LongTensor([ self.y[i] ])
        y = FloatTensor([ self.y[i] ])
        
        #obtain GOP score and normalize it
        #h5_path utterance
        #segment_id number of segment  
        if self.gop:
            gop = self.h5_file[f'{h5_path}/scores_info'][()][segment_id] #this is a single number
            gop = (gop - self.gop_mvn['mean']) / self.gop_mvn['std']
        
        #obtain the VAE embedding (already normalised)
        if self.vae:
            emb = self.emb[i, :]
        
        #obtain LDA topic posterior
        #h5_path utterance
        #segment_id number of segment
        #self.lda_list, self.lda_pz
        if self.lda:
            lda_pz = self.lda_pz[self.lda_list.index(os.path.basename(h5_path)),:]
            lda_pz = (lda_pz - self.lda_mvn['mean']) / self.lda_mvn['std']
        
        
        #generate the feature vector
        feat = np.array([])
        #generate the feature vector
        #the phone tag (already norm), gop score, vae embedding (already normalised), lda posteriors
        if self.gop:
            feat = np.concatenate( [feat, np.array([gop])] ) 
        if self.phseg:
            feat = np.concatenate( [feat, np.array( [ self.phsegref[i] ] ) ] )     
        if self.with_phone:
            feat = np.concatenate( [ feat, np.array( [self.phref[i] ] ) ] )
        if self.with_word:
            feat = np.concatenate( [ feat, np.array( [ self.wdref[i] ] ) ] )      
        if self.with_cntxt:     
            feat = np.concatenate( [ feat, np.array([ self.leftphref[i], self.rightphref[i] ]) ] )   
        if self.preref:
            feat = np.concatenate( [ feat, np.array( [ self.prey[i] ] ) ] )     
        if self.vae:
            feat = np.concatenate( [ feat, emb ] )
        if self.lda:
            feat = np.concatenate( [ feat, lda_pz ] )
            
        feat = FloatTensor( feat )
        
        phone = ph
        speaker = speaker_name
        
        return (feat, y)  
