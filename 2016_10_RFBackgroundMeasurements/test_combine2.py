import matplotlib.pyplot as plt
import numpy as np
import read_dpt
import combine_data
import sys

#generate list of files
file_template = '20161004_trace3_46K_CF1500MHz_BW1800MHz_0{}.dpt'
n_files = 3
files = []

for i in range(1,n_files+1):
    files.append(file_template.format(i))

#read data through list of files
x = [0]*n_files
y = [0]*n_files

for i in range(n_files):
    x[i],y[i] = read_dpt.read_rsa_dpt(files[i])

x_total = np.concatenate(x)
y_total = np.concatenate(y)

plt.plot(x_total,y_total,'.')
plt.show()
#concatenate data
