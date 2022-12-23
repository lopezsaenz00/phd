#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:20:00 2018

@author: Jose Antonio Lopez @ The University of Sheffield
"""
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import ast


#############################################################
def load_txtfile_aslist(file_path):
    #this is for selecting a textfile and
    #returns a list with lists of each line of the file. 

    with open(file_path, 'r') as myfile:
        file_lines =  myfile.read().splitlines()

    return file_lines      
#############################################################

#############################################################
def load_ref_file(ref_file_path):
    #this method loads the reference (annotation file as a 
    #list of lists,
    #Each of which lists the contents of each line of the
    #reference file
    
    with open(ref_file_path, 'r') as ref_file:
        ref_lines = [line.split() for line in ref_file]

    return ref_lines
#############################################################

#############################################################
def find_assessor_ref_file_path(ref_path, assessor, lr=None):
    #this method goes to the directory where the ref
    #files are and builds the path for the file of
    #the corresponding assessor
    #ref_path (string)> the path to the directory of the
    #                    ref files
    #assessor (string)> the assessor required ['a1', a2', ...]
    if lr == None:
        ref_file_path = [f for f in listdir(ref_path) if isfile(join(ref_path, f)) and f.find(assessor)!= -1]
    else:
        ref_file_path = [f for f in listdir(ref_path) if isfile(join(ref_path, f)) and f.find(assessor)!= -1 and f.find(str(lr))!= -1]
    
    return ref_path + ref_file_path[0]
#############################################################


#############################################################
def load_df_from_gzip(filename, index_col=0, sep='\t', compression='gzip'):
	#this method shall load a dataframe previously saved as a gz compressed
	#textfile. Just give the path to the file. This is done just because
	#it's easier to remember that the syntax to load the file exists here.
	#filename (string) : tha path to the file, it should be a csv.gz file
	
	return pd.read_csv(filename, index_col=index_col, sep=sep, compression=compression)
#############################################################

#############################################################
def dict_from_txtfile(file_path):
	#this methods loads a dictionary that has been saved as
	# a text file
	
	return ast.literal_eval(open(file_path, 'r').read())
#############################################################
