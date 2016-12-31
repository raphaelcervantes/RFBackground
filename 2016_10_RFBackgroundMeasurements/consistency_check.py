import matplotlib.pyplot as plt
import numpy as np
import read_dpt
import sys

##### 46K_CF1500MHz_BW1800MHz #######################
file1 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_01.dpt'
file2 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_02.dpt'
file2 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_03.dpt'

plot_title = 'T = 46 K, CF = 1500 MHz, BW = 1800 MHz'
ofile = '46K_CF1500MHz_BW1800MHz_20161004.pdf'


###### 46K_CF3000MHz_BW6000MHz #######################
#
#file1 = '20161004_trace3_46K_CF3000MHz_BW6000MHz_01.dpt'
#file2 = '20161004_trace3_46K_CF3000MHz_BW6000MHz_02.dpt'
#file3 = '20161004_trace3_46K_CF3000MHz_BW6000MHz_03.dpt'
#
#plot_title = 'T = 46 K, CF = 3000 MHz, BW = 6000 MHz'
#ofile = '46K_CF3000MHz_BW6000MHz_20161004.pdf'

###### 110K_CF1500MHz_BW1800MHz #######################
#file1 = '20161005_trace3_110K_CF1500MHz_BW1800MHz_01.dpt'
#file2 = '20161005_trace3_110K_CF1500MHz_BW1800MHz_02.dpt'
#file2 = '20161005_trace3_110K_CF1500MHz_BW1800MHz_03.dpt'
#
#plot_title = 'T = 110 K, CF = 1500 MHz, BW = 1800 MHz'
#ofile = '110K_CF1500MHz_BW1800MHz_20161005.pdf'

######## 110K_CF3000MHz_BW6000MHz #######################
#file1 = '20161004_trace3_46K_CF3000MHz_BW6000MHz_01.dpt'
#file2 = '20161004_trace3_46K_CF3000MHz_BW6000MHz_02.dpt'
#file3 = '20161004_trace3_46K_CF3000MHz_BW6000MHz_03.dpt'
#
#plot_title = 'T = 46 K, CF = 3000 MHz, BW = 6000 MHz'
#ofile = '110K_CF3000MHz_BW6000MHz_20161004.pdf'


ylabel_xcoord = -0.10
x1,y1 = read_dpt.read_rsa_dpt(file1)
x2,y2 = read_dpt.read_rsa_dpt(file2)
x3,y3 = read_dpt.read_rsa_dpt(file2)

if x1[0]==0:
    x1 = x1[1:]
    x2 = x2[1:]
    x3 = x3[1:]
    y1 = y1[1:]
    y2 = y2[1:]
    y3 = y3[1:]

spread = np.ptp(np.vstack((y1,y2,y3)),axis=0)

f, (ax1, ax2) =  plt.subplots(2,1, sharex = True)
ax1.plot(x1,y1, '.', ms = 3, label = 'Run 1')
ax1.plot(x2,y2, '.', ms = 3, label = 'Run 2')
ax1.plot(x3,y3, '.', ms = 3, label = 'Run 3')
ax1.set_ylabel('Power [dBm]')
ax1.set_title(plot_title, size = 'medium')
ax1.yaxis.set_label_coords(ylabel_xcoord, 0.5)
leg = ax1.legend(loc = 'upper right')
ax2.plot(x1,spread, lw = 0.5, marker = '.', ms = 3)
ax2.set_ylabel('Range [dBm]')
ax2.set_xlabel('Frequency [MHz]')
ax2.yaxis.set_label_coords(ylabel_xcoord,0.5)
plt.tight_layout()
plt.savefig(ofile)
plt.close()

def dBm_to_mW(p):
    return 10**(p/10)

y1 = dBm_to_mW(y1)*1e6
y2 = dBm_to_mW(y2)*1e6
y3 = dBm_to_mW(y3)*1e6
ofile = "lin_"+ofile
sig_y = np.std(np.vstack((y1,y2,y3)),axis=0)
ylabel_xcoord = -0.09

f, (ax1, ax2) =  plt.subplots(2,1, sharex = True)
ax1.plot(x1,y1, '.', ms = 3, label = 'Run 1')
ax1.plot(x2,y2, '.', ms = 3, label = 'Run 2')
ax1.plot(x3,y3, '.', ms = 3, label = 'Run 3')
ax1.set_ylabel('Power [nW]')
ax1.set_title(plot_title, size = 'medium')
ax1.yaxis.set_label_coords(ylabel_xcoord, 0.5)
leg = ax1.legend(loc = 'upper right')
ax2.plot(x1,sig_y*1e3, lw = 0.5, marker = '.', ms = 3)
ax2.set_ylabel('$\sigma$ [pW]')
ax2.set_xlabel('Frequency [MHz]')
ax2.yaxis.set_label_coords(ylabel_xcoord,0.5)
plt.tight_layout()
plt.savefig(ofile)
plt.close()

