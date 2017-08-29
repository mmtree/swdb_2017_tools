


import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import math



## First average along all k1, k2 values
Avg_block = np.zeros([9,9])
for k1 = 1:18
    for k2 = 1:18
        Avg_block = Avg_block + W(k1,k2)       # add all values in each W cell along each k1, k2
        
Avg_block = Avg_block/(18*18)      



## Arrange all the distance values in ascending order

dist_vec = np.array([0, 1, 2, 3, 4, math.sqrt(2), math.sqrt(5), math.sqrt(10), math.sqrt(17),
                    2*math.sqrt(2), math.sqrt(13), math.sqrt(20), 3*math.sqrt(2), 5, 4*math.sqrt(2)])

dist_vec = sorted(dist_vec)
print(dist_vec)



## Next average for each distance, group like distances
dist_given_row_col = {}#keys:[(row,col)], values:[distance]

for row in range(0,9):
    for col in range(0,9):
        
        dist_given_row_col.setdefault(math.sqrt((row-4)**2+(col-4)**2),[]).append((row,col))
dist_list=dist_given_row_col.keys
W_dist = []
for dist in dist_list:
    temp = 0.0
    for row,col in dist_given_row_col[dist]:
        temp+=W[row,col]
    temp = temp*(1.0/len(dist_given_row_col[dist]))
    W_dist.append(temp)
plt.plot(dist_list,W_dist,'o-')

