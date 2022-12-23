#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:53:00 2020

@author: Jose Antonio Lopez @ Infirmary Rd

This scripts performs LDA on ITSLANGUAGE using the outputs of
Quantization

"""
import sys
import os
#to grab packages from somewhere else
#sys.path.append( os.environ.get('PYTHON_PACKAGES', None))
from pprint import pprint
import math
import time
import numpy as np
import pandas as pd
#import torch
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
param_names = ['BASE_NAME','VQ_TRAIN_FILELIST', 'VQ_TEST_FILELIST', 'N_TOPICS', 'EPOCHS', 'ETA', 'ALPHA',
 'ITERATIONS' ,'LDA_MODEL_DIR', 'GRAPHS_DIR', 'TOOLS_DIR', 'EXP_DIR', 'N_COMP', 'PZ_DIR', 'TRAIN_SCP', 'TEST_SCP']

parameters = {}
int_parameters = ['N_TOPICS', 'EPOCHS', 'ITERATIONS']
float_parameters = ['ETA', 'ALPHA']

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

################################################################################# 
# Check if theres a previous log file to delete
if os.path.exists(parameters['EXP_DIR']+'INFO_LOG/'+parameters['BASE_NAME']+'.lda_train.log'):
    os.remove(parameters['EXP_DIR']+'INFO_LOG/'+parameters['BASE_NAME']+'.lda_train.log')
    print("Prevoius gensim logfile deleted")
else:
    print("Prevoius gensim logfile does not exist")
################################################################################# 
#    
################################################################################# 
def gen_dictionary(docs_tokenized, min_count = 1):

#this method counts word frequencies in the corpus. It uses the defaultdict
#form the gensim library
# docs_tokenized (list of lists): the utterances with the vq outputs (in string)
# min_count (int) : the minimum fequency of a token to be kept in the corpus

    frequency = defaultdict(int)

    for doc in docs_tokenized:
        for token in doc:
            frequency[token] += 1

    # Only keep words that appear more than once
    docs_tokenized = [[token for token in doc if frequency[token] > min_count] for doc in docs_tokenized]
    #create dictionary
    dictionary = corpora.Dictionary(docs_tokenized)

    return dictionary, docs_tokenized
################################################################################# 

################################################################################# 
def set_min_count(docs_tokenized, min_count = 1):

#this method counts word frequencies in the corpus. It uses the defaultdict
#form the gensim library
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
def vectorize_corpus(docs_tokenized, dictionary):
# it creates the vectorized documents.

# docs_tokenized (list of lists): the utterances with the vq outputs (in string)
# dictionary (collection object): the object holding the Id for every token in
#                                  the corpus. 

	return [dictionary.doc2bow(doc) for doc in docs_tokenized]
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
def train_LDAmodel(tfidf, dictionary, n_topics, epochs, iterations = 50, alpha = 1, eta = 'auto'):
# corpus ({iterable of list of (int, float), scipy.sparse.csc}, optional) – 
#                                  Stream of document vectors or sparse matrix 
#                                   of shape (num_terms, num_documents).
#                                  If not given, the model is left untrained 
#                                   (presumably because you want to call
#                                   update() manually).
# num_topics (int, optional) – The number of requested latent topics to be 
#                               extracted from the training corpus.
# id2word ({dict of (int, str), gensim.corpora.dictionary.Dictionary}) – 
#                                Mapping from word IDs to words. It is used
#                                  to determine the vocabulary size, as well
#                                    as for debugging and topic printing.
# distributed (bool, optional) – Whether distributed computing should be used 
#                                 to accelerate training.
# chunksize (int, optional) – Number of documents to be used in each training chunk.
# passes (int, optional) – Number of passes through the corpus during training.
# update_every (int, optional) – Number of documents to be iterated through 
#                               for each update. Set to 0 for batch learning, > 
#                               1 for online iterative learning.
# alpha ({numpy.ndarray, str}, optional) – A-priori belief on word probability,
# eta ({float, np.array, str}, optional) 
# eval_every (int, optional) – Log perplexity is estimated every that many updates. 
#                              Setting this to one slows down training by ~2x.
# iterations (int, optional) – Maximum number of iterations through the corpus 
#                              when inferring the topic distribution of a corpus.


	alpha = np.repeat(alpha, n_topics)
	temp = dictionary[0]
	
	lda = LdaModel(corpus = tfidf,id2word = dictionary, update_every = 1, alpha = 'auto', num_topics=n_topics, eta = eta, eval_every = 1,   iterations = iterations, passes = epochs )
	
	return lda
#################################################################################


#################################################################################
def print_likelihood_curve(logfile_path, graph_path, graph_basename):
#taken from
# https://stackoverflow.com/questions/37570696/how-to-monitor-convergence-of-gensim-lda-model

	p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
	matches = [p.findall(l) for l in open(logfile_path)]
	matches = [m for m in matches if len(m) > 0]
	tuples = [t[0] for t in matches]
	perplexity = [float(t[1]) for t in tuples]
	likelihood = [float(t[0]) for t in tuples]
	epochs = list(range(0,len(tuples),1))
	plt.plot(epochs,likelihood,c="black")
	plt.ylabel("log likelihood (per-word bound)")
	plt.xlabel("Iterations")
	plt.xticks(np.arange(min(epochs), max(epochs)+1, np.ceil((epochs[-1]+1)/10)))
	plt.title('loglike: '+graph_basename)
	plt.grid()
	plt.savefig(graph_path+graph_basename+".loglike.png")
	print("Loglikelihood curve printed as:\n"+graph_path+graph_basename+".loglike.png")
	plt.close()
	
	plt.plot(epochs,perplexity,c="black")
	plt.ylabel("Perplexity")
	plt.xlabel("Iterations")
	plt.xticks(np.arange(min(epochs), max(epochs)+1, np.ceil((epochs[-1]+1)/10)))
	plt.title('perplexity: '+graph_basename)
	plt.grid()
	plt.savefig(graph_path+graph_basename+".perplexity.png")
	print("Perplexity curve printed as:\n"+graph_path+graph_basename+".perplexity.png")
	plt.close()
	
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


################################################################################
def save_df(dataframe, df_name, return_name = False):
    #if return name is True, the method returns the name of the file.
    
    if not os.path.exists(df_name[:df_name.rfind('/')]):
        os.makedirs(df_name[:df_name.rfind('/')])
    
    dataframe.to_csv(df_name+'.csv.gz', sep='\t', compression='gzip')
    
    if return_name:
        return df_name+'.csv.gz'
################################################################################

################################################################################# 
def save_new_dataset(posteriors, doc_names, data_accents, dictionary, dict_size, epochs, subset, pz_dir, base_name):
	#this method adds the column with the VQ output for every frame and
	#rewrites the dataframe

	#df (pandas Dataframe) : it adds the choice of cluster to each frame
	#choice_cluster (LongTensor) : the cluster with the smallest error.
	
	print("Net dictionary size: "+str(len(dictionary.id2token)))
	n_topics = posteriors.shape[1]
	headers = ['pz'+str(topic) for topic in range(n_topics)]
	posteriors_df = pd.DataFrame(data = posteriors, columns = headers)
	
	#speaker=[ elem[elem.rfind('.', 0, elem.rfind('.'))+1:elem.rfind('_', 0, elem.rfind('_'))] for elem in doc_names]
	speaker = list()
	corpus = list()
	
	for elem in doc_names:
		inds_ = [i for i,c in enumerate(elem) if c=='_']
		inds_dot = [i for i,c in enumerate(elem) if c=='.']
		
		speaker.append( elem[inds_[-2]+1:inds_[-1]] )
		
		corpus.append( elem[inds_dot[-2]+1:inds_[-2]] )
	
	df_for_saving = pd.DataFrame()
	df_for_saving['doc'] = doc_names
	df_for_saving['corpus'] = corpus
	df_for_saving['speaker'] = speaker
	df_for_saving['accent'] = data_accents
	
	df_name = pz_dir+base_name+'.'+subset
	df_for_saving = df_for_saving.join(pd.DataFrame(data = posteriors, columns = headers))
	
	print("LDA posteriors of "+subset+" data saved as:")
	print(save_df(df_for_saving, df_name, return_name = True))
################################################################################# 


################################################################################## 
def load_vq(vq_scp_file, dataset_scp_file):
#this method loads the vq files as lists
# vq_scp_file (string) : the path to the vq scp file

	vq_files = load_txt.load_txtfile_aslist(vq_scp_file)
	#get the accents
	dataset_files=load_txt.load_txtfile_aslist(dataset_scp_file)
	dataset_speakers=list()
	data_IDs = list()
	data_vq = list()
    
	for elem in vq_files:
		name = os.path.basename(elem) 
		speaker=name[ name.rfind('_')+1:name.rfind('.')]
		data_IDs.append(name)
		data_vq.append(load_txt.load_txtfile_aslist(elem))
		name = name.replace('.vq', '')
		dataset_speakers.append(speaker)
		
	
	return data_IDs, data_vq, dataset_speakers
################################################################################## 

#################################################################################
if __name__ == '__main__':

	#load the dataset
	#Data loades is not really required as the data is not been used in
	# minibatches
	print("Load train data:")
	print(parameters['VQ_TRAIN_FILELIST'])
	print("")

	data_IDs, data_vq, data_spkr = load_vq(parameters['VQ_TRAIN_FILELIST'], parameters['TRAIN_SCP'])

	print("Count word frequency and create dictionary.")
	dictionary, docs_tokenized = gen_dictionary(data_vq)
	print("")


	print("Vectorize corpus")
	docs_tokenized = vectorize_corpus(docs_tokenized, dictionary)
	print("")

	print("Generate TFIDF model.") #IT SEEMS LIKE LDA QORKS BETTER WITH INTEGERS
	docs_tfidf = get_tfidf(docs_tokenized)
	print("")

	print("The inverse term frequency of each document.")
	print("Shoudl be similar to the betas used to create the corpus")
	print("Here some documents:")
	count = 0
	for doc in docs_tfidf:
	#for doc in docs_tokenized:
		count += 1
		print(data_IDs[count-1])
		pprint(doc)
		if count > 0:
			print("...")
			break
	print("")
	
	logging.basicConfig(filename =parameters['EXP_DIR']+'INFO_LOG/'+parameters['BASE_NAME']+'.lda_train.log',
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)
                    
	print("Train LDA model.")
	print("Topics: {}".format(parameters['N_TOPICS']))
	print("Per-topic Iterations {}".format(parameters['ITERATIONS']))
	print("Epochs: {}".format(parameters['EPOCHS']))

	lda = train_LDAmodel(docs_tfidf, dictionary, n_topics = parameters['N_TOPICS'], epochs = parameters['EPOCHS'], iterations = parameters['ITERATIONS'])
	
	print("Model trained")
	
	
	print("")
	print("Print likelihood curve.")
	print_likelihood_curve( parameters['EXP_DIR']+'INFO_LOG/'+parameters['BASE_NAME']+'.lda_train.log',parameters['GRAPHS_DIR'], parameters['BASE_NAME'])
	print("")
	print("Log per-word likelihood of training data: "+ str(lda.log_perplexity(docs_tfidf)))
	print("")

	###save the lda model
	print("Save LDA model.")
	temp_file = datapath(parameters['LDA_MODEL_DIR']+parameters['BASE_NAME'])
	lda.save(temp_file)
	
	
	#get posteriors for test data
	print("Load Test Data")
	data_IDs, data_vq, data_spkr = load_vq(parameters['VQ_TEST_FILELIST'], parameters['TEST_SCP'])
	print("")
	print("Create bag of words of test data.")
	#we dont need a new dictionary
	docs_tokenized = set_min_count(data_vq)
	docs_tokenized = vectorize_corpus(docs_tokenized, dictionary)
	print("")

	print("Generate TFIDF model.") #IT SEEMS LIKE LDA QORKS BETTER WITH INTEGERS
	docs_tfidf = get_tfidf(docs_tokenized)
	
	#get posteriors of test data
	print("Log perplexity of test data: "+ str(lda.log_perplexity(docs_tfidf)))
	print("")

	print("Thanks for watching.\n")
