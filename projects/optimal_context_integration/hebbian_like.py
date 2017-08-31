import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import pickle
import scipy as sp
from scipy import signal
import scipy.io as spio
import math
from random import shuffle


import warnings

warnings.filterwarnings('error','invalid value encountered in multiply',RuntimeWarning)



#%matplotlib inline

drive_path = '/home/zhixinl/swdb_2017_tools/projects/optimal_context_integration/'

###########################################################################################
################### LOAD THE SPARCIFIED CONVOLUTED IMAGES FROM PICKLE  ####################
#Get the previously calculated conv_spar_images:
with open('conv_spar_images.pickle', 'rb') as handle:
    conv_spar_images = pickle.load(handle)
########## conv_spar_images[k][i,r,c] is the convoluted image i with filter k at ##########
########## sparcified patch. It is the r-th patch row and c-th patch column.     ##########
########## They are normalized along index of filters k.                         ##########
###########################################################################################
#find out size of conv_spar_image:
M,N = np.shape(conv_spar_images[0][0,:,:])
###########################################################################################
################### Load the W which is the theoritical W.             ####################
with open('W_dict.pickle', 'rb') as handle:
    W_dict = pickle.load(handle)
################### W_dict[(k1,k2)] is the 9-by-9 W matrix.            ####################
###########################################################################################


##############################################################################
##############################################################################
#this two numpy array ccontains the pari of possible center coordinates.
##############################################################################
##############################################################################
center_possible_row_index=[]
center_possible_col_index=[]
#the possible row index is from 4 to M-5
#the possible col index is from 4 to N-5
center_possible_row_index = range(4,M-5+1)*(N-5-4+1)
for col in range(4,N-5+1):
    temp = [col]*(M-5-4+1)
    center_possible_col_index = center_possible_col_index + temp
center_possible_row_index = np.array(center_possible_row_index)
center_possible_col_index = np.array(center_possible_col_index)
##############################################################################
##############################################################################


#make a list of randomly ordered image indexs that go through training set for 10 times.
#rand_learning_order:
learning_order = range(0,137)
rand_learning_order=[]
for i in range(0,10):
    shuffle(learning_order)
    if i>0:
        while learning_order[0]==rand_learning_order[-1]:
            shuffle(learning_order)
    rand_learning_order += learning_order
#have to make sure the two neighboring index is not the same.

############################################################################################
#initialize empty W with 9-by-9 W size (with translational invariance assumption which 
#does not change the result at all for using classical f hebbian learning.)
W_hb = np.zeros([18,18,9,9])
W_hb_result_list = [W_hb]
#save a W_batch_avg for the last term of hebbian like learning update:
W_batch_avg = np.zeros([18,18,9,9])
W_batch_avg_temp = np.zeros([18,18,9,9]) #this is for dump into avg every 10 steps
#define learning rate as ita=0.0001
eta = 0.0001
############################################################################################
#for loop starts from the second picture and compare with the previous image to get delta_W:
for i in range(0,len(rand_learning_order)):
    #########
    #batch number = 10.0
    if np.mod(i,10)==0:##################for each 10 steps make the batch avg of previous 10 W
        W_batch_avg=W_batch_avg_temp/10.0
        W_batch_avg_temp = np.zeros([18,18,9,9])###re-initial the temp W_batch_avg
    #########
    #calculate delta W: delta_W
    delta_W=np.zeros([18,18,9,9])
    f_k1_n1s_history_sum = np.zeros((18,len(center_possible_row_index)))
    f_k2_n2s_history_sum = np.zeros((18,len(center_possible_row_index)))
    for k1 in range(0,18):
        for k2 in range(0,18):
            #this part is for between filter k1 filter k2:
            for delta_n_x in range(-4,5):
                for delta_n_y in range(-4,5):
                    #############################################
                    #the lists (np.array) for center (n2) coordinates:
                    #row: center_possible_row_index
                    #col: center_possible_col_index
                    #the lists (np.array) for free_moving (n_1) coordinates:
                    #row: center_possible_row_index + delta_n_x
                    #col: center_possible_col_index + delta_n_y
                    #############################################
                    ##########################################################################################
                    ##########################################################################################
                    #now we write down the few vectors for nummerators and demomonators in Eq for deltaW_W:
                    ###########################
                    #f_k2_n2s_current:
                    #the vectorized f_k2_n2_s(t) (along x:[center_possible_row_index], y:[center_possible_col_index])
                    f_k2_n2s_current = conv_spar_images[k2][rand_learning_order[i],center_possible_row_index,center_possible_col_index]
                    ###########################
                    #f_k1_n1s_current:
                    #the vectorized f_k1_n1_s(t) (along x:[center_possible_row_index + delta_n_x, y:[center_possible_col_index + delta_n_y])
                    f_k1_n1s_current = conv_spar_images[k1][rand_learning_order[i],center_possible_row_index+delta_n_x, center_possible_col_index+delta_n_y]
                    ###########################
                    #f_k2_n2s_previous:
                    #the vectorized f_k2_n2_s(t-1) (along x:[center_possible_row_index], y:[center_possible_col_index])
                    f_k2_n2s_previous = conv_spar_images[k2][rand_learning_order[i-1],center_possible_row_index,center_possible_col_index]
                    if i-1<0:
                        f_k2_n2s_previous = 0.0*f_k2_n2s_previous
                    ###########################
                    #f_k1_n1s_previous:
                    #the vectorized f_k1_n1_s(t-1) (along x:[center_possible_row_index + delta_n_x, y:[center_possible_col_index + delta_n_y])
                    f_k1_n1s_previous = conv_spar_images[k1][rand_learning_order[i-1],center_possible_row_index+delta_n_x, center_possible_col_index+delta_n_y]
                    if i-1<0:
                        f_k1_n1s_previous = 0.0*f_k1_n1s_previous
                    ###########################
                    #f_k2_n2s_history_sum:
                    #the vectorized f_k2_n2_s_upto(t) (along x:[center_possible_row_index], y:[center_possible_col_index])
                    f_k2_n2s_history_sum[k2,:] = f_k2_n2s_history_sum[k2,:] + f_k2_n2s_current
                    ###########################
                    #f_k1_n1s_history_sum:
                    #the vectorized f_k1_n1_s_upto(t) (along x:[center_possible_row_index + delta_n_x, y:[center_possible_col_index + delta_n_y])
                    f_k1_n1s_history_sum[k1,:] = f_k1_n1s_history_sum[k1,:] + f_k1_n1s_current
                    ###########################
                    ##########################################################################################
                    ##########################################################################################
                    ###########################
                    #Now we compute the delta_W
                    ###########################
                    temp_term_1 = np.divide(f_k1_n1s_current-f_k1_n1s_previous,f_k1_n1s_history_sum[k1,:])
                    temp_term_1[np.isnan(temp_term_1)]=0.0
                    temp_term_2 = np.divide(f_k2_n2s_current-f_k2_n2s_previous,f_k2_n2s_history_sum[k2,:])
                    temp_term_2[np.isnan(temp_term_2)]=0.0
                    delta_W[k1,k2,delta_n_x+4,delta_n_y+4] = np.mean((1.0*(i+1.0)*(i+1.0))*np.multiply(temp_term_1,temp_term_2))-2.0*W_batch_avg[k1,k2,delta_n_x+4,delta_n_y+4]
                    #
                    #
                    ############################
    #########################################################
    #########################################################
    #########################################################
    #########
    #chaneg W based on the delta W:
    W_hb = W_hb + eta*delta_W
    W_hb_result_list.append(W_hb)
    #########
    #add the new W into the dump place: W_batch_avg_temp
    W_batch_avg_temp = W_batch_avg_temp + W_hb
    #########
#############
#############

################ SAVE THE Trajectory of W_hb as a list in W_hb_result_list ##################
#make the last element in this list the theory result:
W_theory = np.zeros([18,18,9,9])
for k1 in range(0,18):
    for k2 in range(0,18):
        W_theory[k1,k2,:,:] = W_dict[(k1,k2)]
W_hb_result_list.append(W_theory)
################ SAVE THE Trajectory of W_hb as a list in W_hb_result_list ##################
with open('W_hb_result_list.pickle', 'wb') as handle:
    pickle.dump(W_hb_result_list, handle)