import numpy as np
import math
import matplotlib.pyplot as plt
from combine_data import concat_data_all_temperatures
import sys
from scipy.ndimage.filters import gaussian_filter
import pdb
import sys
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()


# define temperatures of data taken
desired_temperatures = [40, 60, 70, 80, 90, 100, 110, 120]
#desired_temperatures = [40]
real_temperatures = [48, 58, 68, 71, 90, 100, 110, 120]
#real_temperatures =    [48]
n_files = 31

# import and concatenate all data for all temperatures
data_dir = '2016_09_RFBackgroundMeasurements/'
template = data_dir + '2016_09_23_trace3_{}K_{}.dpt'

x_all_temp, y_all_temp_smooth, xfrag_all_temp, yfrag_all_temp, cf_all_temp = concat_data_all_temperatures(
    template, desired_temperatures, n_files, linearize=True, smooth=True)

T_term = 30  # K
T_amp = 26  # K
T_circ = 30  # K
alpha_circ = 10**(-0.5 / 10)

x_all_temp = x_all_temp[40]
alpha = np.zeros_like(x_all_temp, dtype=float)
sig_alpha = np.zeros_like(x_all_temp, dtype=float)
alpha_m = np.zeros_like(x_all_temp, dtype=float)
realtemp = np.array(real_temperatures)
sys_noise_t = np.zeros((len(realtemp), len(x_all_temp)))
sig_sys_noise_t = np.zeros((len(realtemp), len(x_all_temp)))

# for ind in range(0,len(x_all_temp)): #loop through frequencies
for ind in range(0, 1):  # loop through frequencies
    power_t = np.zeros_like(realtemp, dtype=float)
    for i in range(0, len(realtemp)):  # loop through temperatures
        power_t[i] = y_all_temp_smooth[desired_temperatures[i]][ind]
    z, cov = np.polyfit(realtemp, power_t, 1, cov=True)
    sig_z = np.sqrt(np.diagonal(cov))
    x_int = -z[1] / z[0]
    print(x_int)
    sig_x_int = np.abs(x_int) * \
        ((sig_z[0] / z[0])**2 + (sig_z[1] / z[1])**2)**(1 / 2)
    print(sig_x_int)
    alpha[ind] = -(T_amp + x_int) / (T_term - x_int)
    A = -(T_amp + x_int)
    B = (T_term - x_int)
    sig_alpha[ind] = alpha[ind] * sig_x_int * \
        (1 / A**2 + 1 / B**2 - 2 / (A * B))**(1 / 2)
#    sig_alpha[ind] = np.abs(alpha[ind]/x_int) * sig_x_int*(2)**(1/2)
#    alpha_m[ind] = -(T_circ*(1-alpha_circ) + T_amp +
#        x_int*alpha_circ)/(alpha_circ*(T_term*alpha_circ +
#            T_circ*(1-alpha_circ)- x_int))
print(alpha)
print(sig_alpha)
sys.exit()
for i in range(0, len(realtemp)):  # loop through temperatures
    sys_noise_t[i] = T_term * alpha + realtemp[i] * (1 - alpha) + T_amp
    sig_sys_noise_t[i] = np.abs(sig_alpha) * \
        np.sqrt(T_term**2 + realtemp[i]**2)
plt.figure()
plt.plot(x_all_temp, alpha, rasterized=True)
plt.ylabel('Fractional Transparency')
plt.xlabel('Frequency [MHz]')
plt.tight_layout()
plt.savefig('alpha_v_freq.pdf')
plt.close()
#plt.plot(x_all_temp, alpha_m)

plt.figure()
plt.errorbar(x_all_temp, alpha, sig_alpha, rasterized=True)
plt.ylabel('Fractional Transparency')
plt.xlabel('Frequency [MHz]')
plt.tight_layout()
plt.savefig('alpha_v_freq_w_uncertainties.pdf')
plt.close()

plt.figure()
plt.plot(x_all_temp, sig_alpha / alpha, rasterized=True)
plt.xlabel('Frequency [MHz]')
plt.ylabel(r'$\sigma_{\alpha}/\alpha$')
plt.tight_layout()
plt.savefig('sig_alpha_v_freq.pdf')
plt.close()

plt.figure()
for i in range(0, len(realtemp)):
    plt.plot(
        x_all_temp,
        sys_noise_t[i],
        label='{} K'.format(
            realtemp[i]),
        rasterized=True)
plt.legend(loc='best')
plt.ylabel('System Noise Temperature [K]')
plt.xlabel('Frequency [MHz]')
plt.tight_layout()
plt.savefig('tsys_v_freq.pdf')
plt.close()

plt.figure()
for i in range(0, len(realtemp)):
    plt.errorbar(
        x_all_temp,
        sys_noise_t[i],
        sig_sys_noise_t[i],
        label='{} K'.format(
            realtemp[i]),
        rasterized=True)
plt.legend(loc='best')
plt.ylabel('System Noise Temperature [K]')
plt.xlabel('Frequency [MHz]')
plt.tight_layout()
plt.savefig('tsys_v_freq_w_uncertainties.pdf')
plt.close()

plt.figure()
for i in range(0, len(realtemp)):
    plt.plot(
        x_all_temp,
        sig_sys_noise_t[i] /
        sys_noise_t[i],
        label='{} K'.format(
            realtemp[i]),
        rasterized=True)
plt.legend(loc='best')
plt.ylabel(r'$\sigma_{T_{sys}}/T_{sys}$')
plt.xlabel('Frequency [MHz]')
plt.tight_layout()
plt.savefig('sig_tsys_v_freq.pdf')
plt.close()
sys.exit()

for i in range(0, 31, 5):
    frequency = cf_all_temp[40][i]
    print(frequency)
    ofile = 'power_v_temperature_' + str(int(frequency)) + 'MHz.txt'
    target_file = open(ofile, 'w')
    for destemp, realtemp in zip(desired_temperatures, real_temperatures):
        x = x_all_temp[destemp]
        y = y_all_temp[destemp]
        idx = np.searchsorted(x, frequency, side='left')
        x_interest = x[idx]
        y_interest = y[idx]
        target_file.write(
            '{}\t{}\t{}\n'.format(
                realtemp,
                x_interest,
                y_interest))
        plt.plot(x, y)
    target_file.close()
