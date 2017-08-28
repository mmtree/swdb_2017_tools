import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
#%matplotlib inline



drive_path = '/home/zhixinl/swdb_2017_tools/projects/optimal_context_integration/'
mypath = os.path.join(drive_path,'images/')
image_file_names = [os.path.join(drive_path,'images/',f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
#get list of images
images_list = [(1.0/255)*np.mean(plt.imread(image_file_name),2) for image_file_name in image_file_names]
images_size = [np.shape(image) for image in images_list]
#get all images into the list which has size 321,481:
long_images=[]
long_images_name = []
for i in range(0,200):
    if images_size[i]==(321,481):
        long_images.append(images_list[i])
        long_images_name.append(image_file_names[i])


#get all the filters from the matlab file:
import scipy.io as spio
#loadmat
filters_mat = spio.loadmat(os.path.join(drive_path,'ephys_data_filts.mat'))
filters_origin = [filters_mat['data_filts'][0][i] for i in range(0,18)]
#make filters mean zero:
filters=[f-np.mean(f) for f in filters_origin]



#do the convolution for images with different filters:
from scipy import signal

conv_long_images = {}#{'filter_index':'images'}
conv_long_images_name = {}#{'filter_index':'images_name'}
for i in range(0,len(filters)):
    temp_imag = []
    temp_name = []
    for j in range(0,len(long_images)):
        type(temp_imag)
        temp_imag.append(signal.convolve2d(long_images[j], filters[i], mode='valid'))
        temp_name.append(long_images_name[i])
    conv_long_images[i] = temp_imag
    conv_long_images_name[i] = temp_name
#save the dict
import pickle
conv_result = {'image_names': conv_long_images_name, 'conv_images': conv_long_images }
with open('conv_result.pickle', 'wb') as handle:
    pickle.dump(conv_result, handle)

#Get the previously calculated conv images:
#with open('conv_result.pickle', 'rb') as handle:
#    b = pickle.load(handle)


conv_spar_images = {}
#size of conv_images:
row_num,col_num = np.shape(conv_result['conv_images'][0][0])
# this cord is the sparcifing index for those conv_images:
cord = np.meshgrid(range(0,row_num,7),range(0,col_num,7))
for i in range(0,18):
    temp = []
    for j in range(0,137):
        image_temp = conv_result['conv_images'][i][j]
        temp.append(image_temp[cord[0],cord[1]].T)
    conv_spar_images[i]=temp
with open('conv_spar_images.pickle', 'wb') as handle:
    pickle.dump(conv_spar_images, handle)