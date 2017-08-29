import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import pickle
import scipy as sp
from scipy import signal
import scipy.io as spio
import math


#%matplotlib inline

drive_path = '/home/zhixinl/swdb_2017_tools/projects/optimal_context_integration/'


##############we will then do the following calculation for W for each given brain region.
brain_regions = ['VISp','VISrl','VISl','VISpm','VISal','VISam']
average_receptive_field_sizes = [252.94909598,266.98871983,429.00202749,483.71624697,517.86731824,578.59030484]
##############
##############
##############
##############
##############
##############
##############
##############
##############  
for i in range(0,6):
    brain_region_name = brain_regions[i]
    avg_receptive_field_size = average_receptive_field_sizes[i]
    zoom_factor = np.sqrt(252.94909598/avg_receptive_field_size)
    
    #load the image
    mypath = os.path.join(drive_path,'images/')
    image_file_names = [os.path.join(drive_path,'images/',f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    ###########################GET IMAGES INTO LIST: images_list#####################
    images_list = [(1.0/255)*np.mean(plt.imread(image_file_name),2) for image_file_name in image_file_names]
    images_size = [np.shape(image) for image in images_list]

    #some of these images are 321-by-481 and some are 481-by-321.
    #################GET IMAGES WITH SIZE (321,481) INTO LIST: long_images ###########
    ################### 137 IMAGES FOUND WITH SIZE ROW=321, COL=481 ##################
    long_images_original=[]
    long_images_name = []
    for i in range(0,200):
        if images_size[i]==(321,481):
            long_images_original.append(images_list[i])
            long_images_name.append(image_file_names[i])

    ###################### ZOOM THE LONG IMAGES BASED ON ZOOM ######################
    long_images = [sp.ndimage.interpolation.zoom(img, zoom_factor) for img in long_images_original]

    ################GET FILTERS FROM MATLAB INTO: filters_original##################
    filters_mat = spio.loadmat(os.path.join(drive_path,'ephys_data_filts.mat'))
    filters_original = [filters_mat['data_filts'][0][i] for i in range(0,18)]

    ######################### MAKE EACH FILTER MEAN ZERO ############################
    #make filters mean zero:
    filters=[f-np.mean(f) for f in filters_original]



    ######################################################################################
    ################    WE THEN DO CONVOLUTION ON THE 137 IMAGES        ##################
    ################ WITH ALL 18 MEAN-ZERO FILTERS. IN TOTAL WE WILL    ##################
    ################ GET 137*18 CONVOLUTED IMAGES. SIZE OF EACH IMAGE   ##################
    ################ AFTER CONVOLUTION IS 307-BY-467.                   ##################
    ######################################################################################
    ####### conv_result = {'conv_images': conv_long_images, 'image_names': conv_long_images_name},
    ####### where
    ####### conv_long_images: {filter_index:[LIST OF CONVOLUTED IMAGES]} is the convolution result,
    ####### conv_long_images_name = {}#{'filter_index':[image names ]}.
    #####
    ###
    ##
    #do the convolution for images with different filters:
    conv_long_images = {}#THE CONVOLUTED IMAGES: {'filter_index':[LIST OF CONVOLUTED IMAGES]}
    conv_long_images_name = {}#{'filter_index':[image names ]}
    for i in range(0,len(filters)):
        temp_imag = []
        temp_name = []
        for j in range(0,len(long_images)):
            temp_imag.append(signal.convolve2d(long_images[j], filters[i], mode='valid'))
            temp_name.append(long_images_name[i])
        conv_long_images[i] = temp_imag
        conv_long_images_name[i] = temp_name


    ################ SAVE THE CONVOLUTED IMAGES IN DICTIONARY TO PICKLE ##################
    conv_result = {'image_names': conv_long_images_name, 'conv_images': conv_long_images }
    with open('conv_result_'+brain_region_name+'.pickle', 'wb') as handle:
        pickle.dump(conv_result, handle)


    ###########################################################################################
    ################### LOAD THE CONVOLUTED IMAGES IN DICTIONARY TO PICKLE ####################
    #Get the previously calculated conv images:
    #with open('conv_result.pickle', 'rb') as handle:
    #    conv_result = pickle.load(handle)
    #    conv_long_images_name = conv_result['conv_images']
    #################### the filter k onto image l is                      ####################
    #################### conv_result['conv_images'][k][j] where            ####################
    #################### k =0, 1, ..., 17 and j = 0,1,...,136              ####################
    #################### the size of each convoluted images is: (307, 467).####################
    ###########################################################################################

    ######################################################################################
    ################    WE THEN SPARSELY CHOOSE BY EVERY 7 ROWS AND 7   ##################
    ################ COLUMNS FROM THE CONVOLUTED IMAGES AND SAVE THEM   ##################
    ################ INTO A SMALLER SIZE IMAGE WITH SIZE 44-BY-67.      ##################
    ################ WE END UP HAVING 18*137 CONVOLUTED AND SPARCIFIED  ##################
    ################ IMAGES EACH WITH SIZE 44-BY-67.                    ##################
    ######################################################################################
    conv_spar_images = {}
    (row_num,col_num) = np.shape(conv_result['conv_images'][0][0]) #shape of the convoluted image.

    #define "cord" as: set of row-col coordinates for sparcification of the convoluted image.
    cord = np.meshgrid(range(0,row_num,7),range(0,col_num,7))# (rows:0,8,15,...) (cols:0,8,15,...)

    ############ For-loop through filters i and images j to sparcify them.  ##############
    for i in range(0,18):
        temp = []#list of sparcified convoluted images given filter i.
        for j in range(0,137):
            image_temp = conv_result['conv_images'][i][j]
            #we need to throw away all negatives from image_temp!!! that is the rectify in the paper.
            image_before_throwing_negatives = image_temp[cord[0],cord[1]].T
            image_after_throwing_negatives = np.abs(np.multiply(image_before_throwing_negatives,image_before_throwing_negatives>0))
            temp.append(image_after_throwing_negatives)
        conv_spar_images[i]=temp

    #################### NORMALIZE THE CONVOLUTION ALONG FILTERS AS Eq(6)  ####################
    normalization_factor_matrix = np.zeros(np.shape(conv_spar_images[0]))
    for i in range(0,18):
        conv_spar_images[i] = np.array(conv_spar_images[i])
        normalization_factor_matrix = normalization_factor_matrix + conv_spar_images[i]
    #normalization as in eq 27 and eq 6 as stated in section 4.6
    for i in range(0,18):
        conv_spar_images[i] = conv_spar_images[i]/normalization_factor_matrix
    ########## conv_spar_images[k][i,r,c] is the convoluted image i with filter k at ##########
    ########## sparcified patch. It is the r-th patch row and c-th patch column.     ##########
    ########## They are normalized along index of filters k.                         ##########
    ###########################################################################################
    ########## SAVE THE SPARCIFIED AND CONVOLUTED IMAGES INTO: conv_spar_images      ########## 
    ########## conv_spar_images={FILTER_INDEX : [SPARCIFIED CONVOLUTED IMAGES]}      ##########
    with open('conv_spar_images_'+brain_region_name+'.pickle', 'wb') as handle:
        pickle.dump(conv_spar_images, handle)
    ###########################################################################################


    ###########################################################################################
    ################### LOAD THE SPARCIFIED CONVOLUTED IMAGES FROM PICKLE  ####################
    #Get the previously calculated conv_spar_images:
    #with open('conv_spar_images.pickle', 'rb') as handle:
    #    conv_spar_images = pickle.load(handle)
    ########## conv_spar_images[k][i,r,c] is the convoluted image i with filter k at ##########
    ########## sparcified patch. It is the r-th patch row and c-th patch column.     ##########
    ########## They are normalized along index of filters k.                         ##########
    ###########################################################################################

    def calculate_W_at_a_center(conv_spar_images,k1,k2,center_row_idx,center_col_idx):
        '''calculate a 9-by-9 W matrix for given pair of filters: (k1, k2) where the
        9-by-9 W matrix is masked onto the sparcified images so that center of W matrix is
        at [center_row_idx,center_col_idx] of the sparcified and convoluted images.
        The meaning of element (a,b) in the 9-by-9 W matrix is the Eq. 26 in the paper where
        n2 is the patch corresponding to the position at [center_row_idx,center_col_idx] whereas
        the n1 patch is for the position [a+center_row_idx-4, b+center_col_idx-4].
        ...
        k1 is the index of the first filter
        k2 is the index of the second filter
        the sparcified-and-convoluted-image-index of the center of W is: [center_row_idx,center_col_idx]
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
                W_at_a_center[row_idx-up_left_row,col_idx-up_left_col]=-1.0+avg_fk1fk2/(avg_fk1*avg_fk2)    
        return W_at_a_center

    ###########################################################################################
    ################### Based on the translational invarioance assumption  ####################
    ################### we calculated averaged 9-by-9 W that is averaged   ####################
    ################### on any possible center positions in the sparcified ####################
    ################### and convoluted images.                             ####################
    ###########################################################################################

    #conv_spar_images is of size [M,N] with M = # of row; N = # of col
    (M,N) = np.shape(conv_spar_images[0][0])
    #then the centers of W is only in the following ranges due to boundary:
    #Row Index from 4 to M-5 -> range(4,M-5+1)
    #Col Index from 4 to N-5 -> range(4,N-5+1)
    W_dict = {}

    #save all W for any pairs of filters k1,k2:
    for k1 in range(0,18):
        for k2 in range(0,18):
            #calculate the spatioally averaged W for k1 and k2 with relative n1,n2 position in 9-by-9 matrix
            W_temp = np.zeros([9,9])
            for center_row in range(4,M-5+1):
                for center_col in range(4,N-5+1):
                    W_temp = W_temp + calculate_W_at_a_center(conv_spar_images,k1,k2,center_row,center_col)
            W_dict[(k1,k2)] = (1.0/((M-8)*(N-8)))*W_temp

    ###########################################################################################
    ################### Save the W which is the theoritical W.             ####################
    with open('W_dict_'+brain_region_name+'.pickle', 'wb') as handle:
        pickle.dump(W_dict, handle)
    ################### W_dict[(k1,k2)] is the 9-by-9 W matrix with the    ####################
    ################### translational invariant assumption.                ####################
    ###########################################################################################

    ###########################################################################################
    ################### Load the W which is the theoritical W.             ####################
    #with open('W_dict.pickle', 'rb') as handle:
    #    W_dict = pickle.load(handle)
    ################### W_dict[(k1,k2)] is the 9-by-9 W matrix.            ####################
    ###########################################################################################

    ############################## Calculate the W(\theta)  ###################################
    W_neg_avg = np.zeros((9,9))
    W_pos_avg = np.zeros((9,9))
    for k1 in range(0,18):
        for k2 in range(0,18):
            W_neg_avg += -1.0*np.abs(np.multiply(W_dict[(k1,k2)],W_dict[(k1,k2)]<0))
            W_pos_avg += 1.0*np.abs(np.multiply(W_dict[(k1,k2)],W_dict[(k1,k2)]>0))
    W_neg_avg = W_neg_avg/(18.0**2.0)
    W_pos_avg = W_pos_avg/(18.0**2.0)

    ## Next average for each distance, group like distances, positive W
    dist_given_row_col = {}#keys:[(row,col)], values:[distance]

    for row in range(0,9):
        for col in range(0,9):
            distance_temp = math.sqrt((row-4)**2+(col-4)**2)/zoom_factor
            dist_given_row_col.setdefault(distance_temp,[]).append((row,col))

    dist_list=dist_given_row_col.keys()
    W_dist_pos = []
    for dist in dist_list:
        temp = 0.0
        for row,col in dist_given_row_col[dist]:
            temp+=W_pos_avg[row,col]
        temp = temp*(1.0/len(dist_given_row_col[dist]))
        W_dist_pos.append(temp)
    ## Next average for each distance, group like distances, negative W
    W_dist_neg = []
    for dist in dist_list:
        temp = 0.0
        for row,col in dist_given_row_col[dist]:
            temp+=W_neg_avg[row,col]
        temp = temp*(1.0/len(dist_given_row_col[dist]))
        W_dist_neg.append(temp)
    ########################################################################################
    with open('W_dist_neg_'+brain_region_name+'.pickle', 'wb') as handle:
        pickle.dump(W_dist_neg, handle)
    with open('W_dist_pos_'+brain_region_name+'.pickle', 'wb') as handle:
        pickle.dump(W_dist_pos, handle)
    with open('dist_list_'+brain_region_name+'.pickle', 'wb') as handle:
        pickle.dump(dist_list, handle)
