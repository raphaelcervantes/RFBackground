import numpy as np
import math
import matplotlib.pyplot as plt
from combine_data import concat_data_all_temperatures
import sys
from scipy.ndimage.filters import gaussian_filter
import pdb
import sys
import seaborn as sns
from uncertainties import ufloat
from uncertainties import unumpy

sns.set(style='ticks', palette='Set2')
sns.despine()

# define temperatures of data taken
desired_temperatures = [40, 60, 70, 80, 90, 100, 110, 120]
real_temperatures = [48, 58, 68, 71, 90, 100, 110, 120]
n_files = 31

# import and concatenate all data for all temperatures
data_dir = '2016_09_RFBackgroundMeasurements/'
template = data_dir + '2016_09_23_trace3_{}K_{}.dpt'

x_all_temp, y_all_temp_smooth, xfrag_all_temp, yfrag_all_temp, cf_all_temp = concat_data_all_temperatures(
    template, desired_temperatures, n_files, linearize=True, smooth=True)

T_term = ufloat(30, 0)  # K
T_amp = ufloat(26, 0)  # K
T_circ = ufloat(30, 0)  # K
alpha_circ = ufloat(10**(-0.5 / 10), 0)

x_all_temp = x_all_temp[40]
alpha = unumpy.uarray([0] * len(x_all_temp), [0] * len(x_all_temp))
#alpha_m = np.zeros_like(x_all_temp, dtype = float)
realtemp = np.array(real_temperatures)
sys_noise_t = np.zeros((len(realtemp), len(x_all_temp)))
sig_sys_noise_t = np.zeros((len(realtemp), len(x_all_temp)))


for ind in range(0, len(x_all_temp)):  # loop through frequencies
    power_t = np.zeros_like(realtemp, dtype=float)
    for i in range(0, len(realtemp)):  # loop through temperatures
        power_t[i] = y_all_temp_smooth[desired_temperatures[i]][ind]
    z, cov = np.polyfit(realtemp, power_t, 1, cov=True)
    sig_z = np.sqrt(np.diagonal(cov))
    z = unumpy.uarray(z, sig_z)
    x_int = -z[1] / z[0]
    alpha[ind] = (-(T_amp + x_int) / (T_term - x_int))
#    alpha_m[ind] = -(T_circ*(1-alpha_circ) + T_amp +
#        x_int*alpha_circ)/(alpha_circ*(T_term*alpha_circ +
#            T_circ*(1-alpha_circ)- x_int))
for i in range(0, len(realtemp)):  # loop through temperatures
    tsys = T_term * alpha + realtemp[i] * (1 - alpha) + T_amp
    sys_noise_t[i] = unumpy.nominal_values(tsys)
    sig_sys_noise_t[i] = unumpy.std_devs(tsys)

sig_alpha = unumpy.std_devs(alpha)
alpha = unumpy.nominal_values(alpha)

plt.figure()
plt.plot(x_all_temp, alpha, rasterized=True)
plt.ylabel('Fractional Transparency')
plt.xlabel('Frequency [MHz]')
plt.tight_layout()
plt.savefig('alpha_v_freq_u.pdf')
plt.close()
#plt.plot(x_all_temp, alpha_m)

plt.figure()
plt.errorbar(x_all_temp, alpha, sig_alpha, rasterized=True)
plt.ylabel('Fractional Transparency')
plt.xlabel('Frequency [MHz]')
plt.tight_layout()
plt.savefig('alpha_v_freq_w_uncertainties_u.pdf')
plt.close()

plt.figure()
plt.plot(x_all_temp, sig_alpha / alpha, rasterized=True)
plt.xlabel('Frequency [MHz]')
plt.ylabel(r'$\sigma_{\alpha}/\alpha$')
plt.tight_layout()
plt.savefig('sig_alpha_v_freq_u.pdf')
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
plt.savefig('tsys_v_freq_u.pdf')
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
plt.savefig('tsys_v_freq_w_uncertainties_u.pdf')
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
plt.savefig('sig_tsys_v_freq_u.pdf')
plt.close()


