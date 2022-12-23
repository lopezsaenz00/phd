#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:35:00 2020

@author: Jose Antonio Lopez @ The University of Sheffield

This scripts loads a LDA model and gets the posterior for files per speaker

"""
import sys
import os
from pprint import pprint
import math
import time
import numpy as np
import pandas as pd
import random
import logging
import re
#import libraries for graphics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D

#gensim for text preprocessing and lda
import gensim.parsing.preprocessing as preprocessing
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim.models import LdaModel
from gensim.models.callbacks import PerplexityMetric
from gensim.test.utils import datapath

    
#NOTE***labels path = features paths
#import parameters
param_names = ['N_COMP', 'N_TOPICS', 'GRAPHS_DIR' , 'BASE_NAME', 'TOOLS_DIR', 'PZ_DIR', 'LDA_POST_SCRIPT', 'VQ_FILELIST', 'MODEL_NAME', 'DATA_SCP', 'SUBSET', 'EPOCHS' ]

parameters = {}
int_parameters = []
float_parameters = []

for name in param_names:
	variable = os.environ.get(name, None)
	if not variable:
		raise ValueError('You must have "'+name+'" variable')
	if name.lower().find('dir')!=-1 and variable[-1]!='/' and name.lower().find('file')==-1:
		variable = variable+'/'
	if name in int_parameters:
		variable = int(variable)
	if name in float_parameters:
		variable = float(variable)
	parameters[name] = variable
	print(name+' : '+str(variable))
    
#import custom modules
sys.path.insert(0, parameters['TOOLS_DIR'])
import load_txt
import save_txt
import load_dataset
import gen_graphs


###for fixing random seeds
import random


################################################################################## 
def load_vq(vq_scp_file, dataset_scp_file):
#this method loads the vq files as lists
# vq_scp_file (string) : the path to the vq scp file

	vq_files = load_txt.load_txtfile_aslist(vq_scp_file)
	#get the accents
	dataset_files=load_txt.load_txtfile_aslist(dataset_scp_file)
	dataset_speakers=list()
	speaker_per_utt = list()
	data_IDs = list()
	data_vq = list()
	
	for line in vq_files:
		name = os.path.basename(line)
		speaker=name[ name.rfind('_')+1:name.rfind('.')]
		speaker_per_utt.append(speaker)
		data_IDs.append(name)
		data_vq.append(load_txt.load_txtfile_aslist(line))
		if not(speaker in dataset_speakers):
			dataset_speakers.append(speaker)
		
	
	return data_IDs, data_vq, speaker_per_utt, dataset_speakers
################################################################################## 

#################################################################################
def vectorize_corpus(docs_tokenized, dictionary):
# it creates the vectorized documents.

# docs_tokenized (list of lists): the utterances with the vq outputs (in string)
# dictionary (collection object): the object holding the Id for every token in
#                                  the corpus. 

	return [dictionary.doc2bow(doc) for doc in docs_tokenized]
#################################################################################

################################################################################# 
def set_min_count(docs_tokenized, min_count = 1):

#this method counts word frequencies in the corpus. It uses the defaultdict
#form the gensim libraryspeaker_per_utt
# docs_tokenized (list of lists): the utterances with the vq outputs (in string)
# min_count (int) : the minimum fequency of a token to be kept in the corpus

    frequency = defaultdict(int)

    for doc in docs_tokenized:
        for token in doc:
            frequency[token] += 1

    # Only keep words that appear more than once
    docs_tokenized = [[token for token in doc if frequency[token] > min_count] for doc in docs_tokenized]

    return docs_tokenized
################################################################################# 

#################################################################################
def get_tfidf(docs_tokenized):
# it creates the tfidf vectors of the corpus.

# docs_tokenized (list of lists): the utterances with the (token id, freq) tuple
# dictionary (collection object): the object holding the Id for every token in
#                                  the corpus. 

	tfidf = models.TfidfModel(docs_tokenized)
	
	return tfidf[docs_tokenized]
#################################################################################

#################################################################################
def get_topic_posteriors(lda, docs_tokenized):
# this method computes the topic posteriors from a bag of words object and
# returns then in an array of shape (docs, topics)
# 
# lda (gensim.models.ldamodel) : the lda model trained.
# docs_tokenized (lists) : a list with the bag of words (count)of the documents

	topics = lda.num_topics
	n_docs = len(docs_tokenized)
	
	doc_pz = np.zeros((n_docs,topics))
	
	#turn topic posteriors into an array
	for doc in range(n_docs):
	
		pz_list = lda[docs_tokenized[doc]]
		
		for topic in range(len(pz_list)):

			doc_pz[doc][pz_list[topic][0]] = pz_list[topic][1]
			
		if len(pz_list) != topics:
			doc_pz[doc,:] = np.where(doc_pz[doc,:]!=0, doc_pz[doc,:], (1.0-doc_pz[doc,:].sum())/(topics-len(pz_list)))
			
	return doc_pz		
#################################################################################


################################################################################# 
def save_new_dataset(posteriors, doc_names, speaker_labels, dictionary, dict_size, epochs, subset, pz_dir, base_name):
	#this method adds the column with the VQ output for every frame and
	#rewrites the dataframe

	#df (pandas Dataframe) : it adds the choice of cluster to each frame
	#choice_cluster (LongTensor) : the cluster with the smallest error.
	
	print("Net dictionary size: "+str(len(dictionary.id2token)))
	n_topics = posteriors.shape[1]
	headers = ['pz'+str(topic) for topic in range(n_topics)]
	posteriors_df = pd.DataFrame(data = posteriors, columns = headers)
	
	
	df_for_saving = pd.DataFrame()
	df_for_saving['doc'] = doc_names
	df_for_saving['speaker'] = speaker_labels
	
	df_name = pz_dir+base_name+'.'+subset
	df_for_saving = df_for_saving.join(pd.DataFrame(data = posteriors, columns = headers))
	
	print("LDA posteriors of "+subset+" data saved as:")
	print(save_df(df_for_saving, df_name, return_name = True))
################################################################################# 

################################################################################
def save_df(dataframe, df_name, return_name = False):
    #if return name is True, the method returns the name of the file.
    
    if not os.path.exists(df_name[:df_name.rfind('/')]):
        os.makedirs(df_name[:df_name.rfind('/')])
    
    #dataframe.to_csv(df_name+'.csv.gz', sep='\t', compression='gzip')
    dataframe.to_csv(df_name+'.csv.gz', sep=',', compression='gzip')
    
    if return_name:
        return df_name+'.csv.gz'
################################################################################


#################################################################################
if __name__ == '__main__':

	print("Load lda model and dictionary:")
	print(parameters['MODEL_NAME'])
	temp_file = datapath(parameters['MODEL_NAME'])
	lda = LdaModel.load(temp_file)
	
	dictionary = lda.id2word

	#load the dataset
	#Data loades is not really required as the data is not been used in
	# minibatches
	print("Load the data:")
	print(parameters['VQ_FILELIST'])
	print("")
	
	data_IDs, data_vq, speaker_labels, speakers = load_vq(parameters['VQ_FILELIST'], parameters['DATA_SCP'])
    
	
	##########vectorize corpus
	print("Create bag of words of the data.")
	data_vq = set_min_count(data_vq)
	data_bow = vectorize_corpus(data_vq, dictionary)
	
	print("Create tfidf of the data.")
	data_tfidf = get_tfidf(data_bow)
	
	data_pz = get_topic_posteriors(lda, data_tfidf)
	
	save_new_dataset(data_pz, data_IDs, speaker_labels, dictionary, int(parameters['N_COMP']), parameters['EPOCHS'], parameters['SUBSET'], parameters['PZ_DIR'], parameters['BASE_NAME'])
	print("Thanks for watching.\n")
