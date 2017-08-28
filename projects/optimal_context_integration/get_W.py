import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import pickle
from scipy import signal
#%matplotlib inline



drive_path = '/home/zhixinl/swdb_2017_tools/projects/optimal_context_integration/'

#Get the previously calculated conv images:
with open('conv_spar_images.pickle', 'rb') as handle:
    conv_spar_images = pickle.load(handle)
for i in range(0,18):
    conv_spar_images[i] = np.array(conv_spar_images[i])
#the conv_spar_images[k][i] is the convoluted and sparcified image from 
#filter k and image i
#conv_spar_images[k][i]
# in this research we only calculate W for size 9 by 9.
#if each conv_spar_images is of size [M,N] M row N col
(M,N) = np.shape(conv_spar_images[0][0])
#then the centers of W is from:
#Row Index from 4 to M-5 -> range(4,M-5+1)
#Col Index from 4 to N-5 -> range(4,N-5+1)

def calculate_W_at_a_center(conv_spar_images,k1,k2,center_row_idx,center_col_idx):
    '''calculate the W for given k1, k2 for one patch at [center_row_idx,center_col_idx]
    and the other patch from the 9-by-9 grid around the center.
    k1 is the index of the first filter
    k2 is the index of the second filter
    the index of the center of W is: [center_row_idx,center_col_idx]
    '''
    #conv_spar_images[k][i]
    # in this research we only calculate W for size 9 by 9.
    #for the notation in the paper where n1 and n2 are index of two patches, we are 
    #going to say that the n1 is at (row+\Delta row, col+\Delta col) and the n2 is at the center.
    #this means that the n1 for the left upper cornor has [\Delta row < 0, \Delta col < 0]
    W_at_a_center=np.zeros([9,9])
    #W_at_a_center[0,0] is the W value at upper left cornor.
    #the index for the up left corner is: [center_row_idx-4,center_col_idx-4]
    #the coordinate we use is the row and col is [0,0] at upper left corner.
    up_left_row = center_row_idx-4
    up_left_col = center_col_idx-4
    for row_idx in range(up_left_row,up_left_row+9):
        for col_idx in range(up_left_col,up_left_col+9):
            avg_fk1fk2 = np.mean(conv_spar_images[k1][:,row_idx,col_idx]*conv_spar_images[k2][:,center_row_idx,center_col_idx])
            avg_fk1 = np.mean(conv_spar_images[k1][:,row_idx,col_idx])
            avg_fk2 = np.mean(conv_spar_images[k2][:,center_row_idx,center_col_idx])
            ##########
            W_at_a_center[up_left_row-row_idx,up_left_col-col_idx]=-1.0+avg_fk1fk2/(avg_fk1*avg_fk2)    
    return W_at_a_center

#if each conv_spar_images is of size [M,N] M row N col
(M,N) = np.shape(conv_spar_images[0][0])
#then the centers of W is from:
#Row Index from 4 to M-5 -> range(4,M-5+1)
#Col Index from 4 to N-5 -> range(4,N-5+1)
W_dict = {}
for k1 in range(0,18):
    for k2 in range(0,18):
        W_temp = np.zeros([9,9])
        for center_row in range(4,M-5+1):
            for center_col in range(4,N-5+1):
                W_temp = W_temp + calculate_W_at_a_center(conv_spar_images,k1,k2,center_row,center_col)
        W_dict[(k1,k2)] = (1.0/(M-8)*(N-8))*W_temp
#save result W:
with open('W_dict.pickle', 'wb') as handle:
    pickle.dump(W_dict, handle)