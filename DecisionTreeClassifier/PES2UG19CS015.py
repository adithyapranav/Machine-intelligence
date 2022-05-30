'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random
import math

'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
	# TODO
	values = list(df.iloc[:, -1:].value_counts())
	entropy = 0
	for i in range(len(values)):
		fract = values[i]/sum(values)
		if(fract != 0):
			entropy = entropy - (fract*np.log2(fract))
	return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
	# TODO
	attribute_values = df[attribute].values
	uav = np.unique(attribute_values)
	rows = df.shape[0]
	avg_of_attribute = 0
	for current_value in uav:
		df_slice = df[df[attribute] == current_value]
		target = df_slice[[df_slice.columns[-1]]].values
		_, counts = np.unique(target, return_counts=True)
		total_count = np.sum(counts)
		entropy = 0
		for freq in counts:
			temp = freq/total_count
			if temp != 0:
				entropy -= temp*np.log2(temp)
		avg_of_attribute += entropy*(np.sum(counts)/rows)
	return(abs(avg_of_attribute))
	
'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
	# TODO
	information_gain = 0
	entropy_of_attribute = get_avg_info_of_attribute(df, attribute)
	entropy_of_dataset = get_entropy_of_dataset(df)
	information_gain = entropy_of_dataset - entropy_of_attribute
	return information_gain

#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
	'''Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected
	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''
	# TODO
	d={}
	col= df.columns
	col=col[:-1]
	for n in range(len(col)):
		attribute= col[n]
		x=get_information_gain(df,attribute)
		d[col[n]]=x
		max_key_value=max(d, key= lambda x: d[x])
		tup=[]
		tup.append(d)
		tup.append(max_key_value)
		tup=tuple(tup)
	return tup
	pass