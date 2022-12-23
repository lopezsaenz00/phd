#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:20:00 2018

@author: Jose Antonio Lopez @ The University of Sheffield

These methods create text files used in the assessor model experiments
on the INA dataset
"""
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import json



#############################################################
def write_ref_file(new_ref_filename, new_ref_lines,new_ref_path ):
    #this method creates a textfile with the same format of
    #the ref files used in the ITS language project (MLF type file).
    
    #new_ref_filename (string) > the name for the file (not path)
    #new_ref_path (string) > the path for saving the new ref file
    #new_ref_lines (list of lists) > Each list contains the elements
    #                   for each line of the new ref file.
    #                   each element in each line will be joint 
    #                   with a '\t' char
    #output:
    # the complete path of the new file (string)
    
    #if the directory doesnt exist, make it
    if not os.path.exists(new_ref_path):
        os.makedirs(new_ref_path)
        
    print(new_ref_path+ ' exists.')
            
    with open(new_ref_path+new_ref_filename, 'w') as f:
        
        for i in range(len(new_ref_lines)):
            
            if len(new_ref_lines[i])==1:
            
                f.write(new_ref_lines[i][0]+'\n')       
            else:
                f.write('\t'.join(new_ref_lines[i]) + '\n')

    f.close()   
    
    print('Ref file saved as: '+ new_ref_path+new_ref_filename)
#############################################################

#############################################################
def write_train_metrics_3columns(filename,  metric_train, metric_test, metricname, csvfile=False):
    #this method write the data of a trainig metric from training
    #filename (string) > the name of the file
    #metric_train/metric_test (np.array) > both a float array 
    #                            with the values of the metric.
    #                            make them same length/shape
    #                          The array is 1D with length=Epoch
    #metricname (string) > the name of the metric saved
    #                          [acc, loss, ...]
    
    #csvfile determines the separator for the file
    #csv file could be useful to modify graphs in frontOffice
    
    if not os.path.exists(filename[:filename.rfind('/')+1]):
        os.makedirs(filename[:filename.rfind('/')+1])
    
    if csvfile:
        sep = ','
        filename = filename+'.csv'
    else:
        sep = '\t'

    with open(filename, 'w') as f:

        f.write(sep.join(['Epoch', 'Train_'+metricname, 'Test_'+metricname]) + '\n')      
        for li in range(metric_train.shape[0]):
            f.write(sep.join([str(li+1),str(metric_train[li]),str(metric_test[li])]) + '\n')

    f.close()
    
    print('Training '+metricname+' report saved as:')
    print(filename)
#############################################################


#############################################################
def write_prediction_file(prediction_filename, prediction_names, prediction_df, assessor_lines, prediction_path):
    #this method replicates the structure of the
    #annotator reference file (MLF like)
    #with the predictions
    
    #check the format of the path
    if prediction_path[-1]!='/':
        prediction_path = prediction_path+'/'
    
    #find the index of all the utterances name
    utt_index_ref = list()
    
    #select the indeces of the name of each utterance.
    for i in range(len(assessor_lines)):
        if len(assessor_lines[i]) == 1 and assessor_lines[i][0] != '.' and assessor_lines[i][0] != '#!MLF!#':
            utt_index_ref.append(i)
            
    #if the directory doesnt exist, make it
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    with open(prediction_path+prediction_filename, 'w') as f:
        
        for ut in range(len(utt_index_ref)):
            
            #if the given utterance is contained in the predicted files
            if assessor_lines[utt_index_ref[ut]][0] in prediction_names:
                #find the index of the utterance in the prediction_names list
                prediction_index = prediction_names.index(assessor_lines[utt_index_ref[ut]][0])
                current_pred_df = prediction_df[prediction_index].values.tolist()
                
                f.write(assessor_lines[utt_index_ref[ut]][0]+'\n')
                
                for li in range(len(current_pred_df)):
                    f.write('\t'.join(current_pred_df[li]) + '\n')

                f.write('.\n')
                
    print("The new ref file has been saved as:")
    print(prediction_path+prediction_filename)
#############################################################

################################################################################
def save_tensor2txt(tensor,tensor_name, return_name = False):
    
    #if return name is True, the method returns the name of the file.
    tensors_path = tensor_name[:tensor_name.rfind('/')+1]
    if not os.path.exists(tensors_path):
        os.makedirs(tensors_path)
        
    tensor_name = tensor_name+'.gz'

#    np.savetxt(tensor_name.encode('ascii','ignore'), tensor.cpu().numpy(), delimiter='\t', fmt='%.8f')
    np.savetxt(tensor_name, tensor.cpu().numpy(), delimiter='\t', fmt='%.8f') # python3 doesnt care about string encoding
    
    if return_name:
        return tensor_name
################################################################################ 


################################################################################
def save_df(dataframe, tensors_path,dataframe_name, return_name = False):
    #if return name is True, the method returns the name of the file.
    
    if not os.path.exists(tensors_path):
        os.makedirs(tensors_path)
    
    dataframe_name = tensors_path+dataframe_name+'.csv.gz'
    
    dataframe.to_csv(dataframe_name, sep='\t', compression='gzip')
    
    print("Dataframe saved as text as:")
    print(dataframe_name)
    
    if return_name:
        return dataframe_name
################################################################################


################################################################################ 
def save_numpy2txt(array,arr_name, return_name = False):
    
    #if return name is True, the method returns the name of the file.
    arr_path = arr_name[:arr_name.rfind('/')+1]
    if not os.path.exists(arr_path):
        os.makedirs(arr_path)
        
    arr_name = arr_name+'.gz'

#    np.savetxt(tensor_name.encode('ascii','ignore'), tensor.cpu().numpy(), delimiter='\t', fmt='%.8f')
    np.savetxt(arr_name, array, delimiter='\t', fmt='%.8f') # python3 doesnt care about string encoding
    
    if return_name:
        return arr_name
################################################################################  


################################################################################
def df2txtfile(dataframe, save_path,dataframe_name, sep='\t', float_format = '%.4f'):
	#this uses the df.to_csv() from pandas to save a dataframe into a textfile
	#save_path (string) : the directory to save the file into
	
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	dataframe_name = save_path+dataframe_name
	
	dataframe.to_csv(dataframe_name, sep= sep, float_format=float_format)
	
	print("Dataframe saved as text as:")
	print(dataframe_name)	
################################################################################

################################################################################
def df2wiki(dataframe, save_path, float_format = '.4f'):
	#this writes a table in a format that can be pasted into the wiki
	#save_path (string) : the directory to save the file into
	
		
	if '.wiki' not in save_path:
		if save_path[-1] == '.':
			dataframe_name = save_path + 'wiki'
		else:
			dataframe_name = save_path + '.wiki'
	
	headers_list = list(dataframe)
	
	headers_list = ["||" + header for header in headers_list]
	headers_list = [header+" " for header in headers_list]
	headers_list[-1] = headers_list[-1]+"||"
	headers = np.array(headers_list)
	values = dataframe.values
	
	
	float_format = "{:"+float_format+"}"
	
	list_of_lists = list()
	for i in range(values.shape[0]):
		row_list = list()
		for j in range(values.shape[1]):
			if isinstance(values[i][j], float):
				row_list.append( "||"+float_format.format(values[i][j])+" " )
			else:
				row_list.append( "||"+str(values[i][j])+" ")
			if j == range(values.shape[1])[-1]:
				row_list[-1] = row_list[-1]+"||"
		
		list_of_lists.append(row_list)		
	
	#The none is for fixing the dimensions and make them able to concatenate
	values = np.array(list_of_lists)
	values = np.concatenate((headers[None,:], values), axis = 0)
	
	np.savetxt(dataframe_name, values, fmt='%s')
	
	print("DataFrame saved in a wiki-tabular format as:")
	print(dataframe_name)
################################################################################

################################################################################
def stringlist2file(list_of_strings, save_path):
#this file just writes a list of strings as a textfile with each element of the list
#as a line
#	*to do: arrange formats for floating points and integers


	with open(save_path, 'w') as f:
		for item in list_of_strings:
			f.write("%s\n" % item)
      
	print("List saved as: "+save_path )
################################################################################

################################################################################
def dict2txtfile(dictionary, save_path):
#this saves a dictionary objext as a text file

	with open(save_path, 'w') as file:
		file.write(json.dumps(dictionary)) # use `json.loads` to do the reverse

	print("Dictionary saved as: "+save_path )
		
