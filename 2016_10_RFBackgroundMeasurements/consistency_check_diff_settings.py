import matplotlib.pyplot as plt
import numpy as np
import read_dpt
import sys

###### 46K #######################
#file1 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_01.dpt'
#file2 = '20161004_trace3_46K_CF3000MHz_BW6000MHz_02.dpt'
#plot_title = 'T = 46 K'
#ofile = '46K_20161004.pdf'


###### 110K_CF1500MHz_BW1800MHz #######################
file1 = '20161005_trace3_110K_CF1500MHz_BW1800MHz_01.dpt'
file2 = '20161005_trace3_110K_CF3000MHz_BW6000MHz_02.dpt'
plot_title = 'T = 110 K'
ofile = '110K_20161005.pdf'

ylabel_xcoord = -0.10
x1,y1 = read_dpt.read_rsa_dpt(file1)
x2,y2 = read_dpt.read_rsa_dpt(file2)

rel_index = np.where((x2>x1[0]) & (x2<x1[-1]))
x2 = x2[rel_index]
y2 = y2[rel_index]

def dBm_to_mW(p):
    return 10**(p/10)

y1_lin = dBm_to_mW(y1)*1e9
y2_lin = dBm_to_mW(y2)*1e9

f, (ax1, ax2) =  plt.subplots(2,1, sharex = True)
ax1.plot(x1,y1, '.', ms = 3, label = 'CF = 1500 MHz, BW = 1800 MHz')
ax1.plot(x2,y2, '.', ms = 3, label = 'CF = 3000 MHz, BW = 6000 MHz')
ax1.set_ylabel('Power [dBm]')
ax1.set_title(plot_title, size = 'medium')
ax1.yaxis.set_label_coords(ylabel_xcoord, 0.5)
leg = ax1.legend(loc = 'upper right')
ax2.plot(x1,y1_lin, '.', ms = 3, label = 'CF = 1500 MHz, BW = 1800 MHz')
ax2.plot(x2,y2_lin, '.', ms = 3, label = 'CF = 3000 MHz, BW = 6000 MHz')
ax2.set_ylabel('Power [fW]')
ax2.yaxis.set_label_coords(ylabel_xcoord, 0.5)
leg = ax1.legend(loc = 'upper right')
plt.tight_layout()
plt.savefig(ofile)
plt.close()


