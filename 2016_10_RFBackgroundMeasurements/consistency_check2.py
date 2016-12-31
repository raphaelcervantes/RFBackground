import matplotlib.pyplot as plt
import numpy as np
import read_dpt
import sys

file1 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_01.dpt'
file2 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_02.dpt'
file2 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_03.dpt'

plot_title = 'T = 46 K, CF = 1500 MHz, BW = 1800 MHz'
ofile = '46K_CF1500MHz_BW1800MHz_20161004.pdf'
ylabel_xcoord = -0.10

x1,y1 = read_dpt.read_rsa_dpt(file1)
x2,y2 = read_dpt.read_rsa_dpt(file2)
x3,y3 = read_dpt.read_rsa_dpt(file2)

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

