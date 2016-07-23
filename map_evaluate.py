#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import math
import copy
from scipy import spatial
import math
import numpy as np
import pickle

if __name__ == "__main__":
	if sys.argv:
		ans_list = []
		predict_list =[]
		#open ans
		with open('AssessmentTrainSet.txt') as ans_set:
			current_query_ans=[]
			for line in ans_set:	
							
				if line in ['\n', '\r\n']:
					ans_list.append(current_query_ans)
					current_query_ans=[]

				else:
					if line.split()[0] != 'Query':
						current_query_ans.append(line.split()[0])
		#open predict
		with open('ResultsTrainSet.txt') as predict_set:
			current_query_ans=[]
			for line in predict_set:
							
				if line in ['\n', '\r\n']:
					predict_list.append(current_query_ans)
					current_query_ans=[]

				else:
					if line.split()[0] != 'Query':
						current_query_ans.append(line.split()[0])

		result_precision = []
		result_recall =[]
		result_averge =[]

		#cuont precision 
		for query_now in range(len(ans_list)):
			
			current_recall = []
			current_precision = []
			ans_num = len(ans_list[query_now])
			predict_num = len(predict_list[query_now])

			current_match_num = 0
			
			for entity_in_predict in range(len(predict_list[query_now])):
				if predict_list[query_now][entity_in_predict] in ans_list[query_now]:
					current_match_num = current_match_num +1
					current_precision.append((current_match_num)/(entity_in_predict+1.0))
					
					#print 'Precision:'+ str(round((current_match_num*100)/(entity_in_predict+1.0),3)) + '%' +'\t Recall:' +str(round((current_match_num*100)/(ans_num+0.0),3))+ '%'
			sum_now = sum(current_precision)
			average_mean_now = sum_now/ans_num
			print 'Curreny query precision:' +str(average_mean_now)
			print '============Query'+str(query_now+1)+'============'
			#result_precision.append(current_precision)
			#result_recall.append(current_recall)
			result_averge.append(average_mean_now)
		print 'MAP:'+ str(sum(result_averge)/len(ans_list))