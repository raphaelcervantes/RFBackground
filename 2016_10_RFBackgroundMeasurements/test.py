import matplotlib.pyplot as plt
import numpy as np
import read_dpt
import sys

file1 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_01.dpt'
file2 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_02.dpt'
file2 = '20161004_trace3_46K_CF1500MHz_BW1800MHz_03.dpt'
x1,y1 = read_dpt.read_rsa_dpt(file1)
x2,y2 = read_dpt.read_rsa_dpt(file2)
x3,y3 = read_dpt.read_rsa_dpt(file2)


print(np.ptp(np.vstack((y1,y2,y3)),axis=0))
sys.exit()
def fft(x,y, freq_max = 0.02):
    fft_y = np.fft.rfft(y)
    psd = np.abs(fft_y)
    fft_xmin = 1/(x[-1]-x[0])
    fft_xmax = 1/2*1/(x[1]-x[0])
    freq = np.fft.rfftfreq(len(y), x[1]-x[0])
    ind_relevant=np.where((freq>fft_xmin) & (freq < freq_max))
    return (freq[ind_relevant],psd[ind_relevant])

#freq1, psd1 = fft(x1,y1)
#freq2, psd2 = fft(x2,y2)
print(np.array_equal(y1,y2))
print(np.array_equal(y1,y3))
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)
plt.show()
sys.exit()
#plt.plot(x2,y2, 'bx')
#
#plt.plot(1/freq1,psd1, '--.')
#plt.plot(1/freq2,psd2, '--.')
#plt.show()
