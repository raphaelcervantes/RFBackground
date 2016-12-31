import numpy as np
import matplotlib.pyplot as plt
from combine_data import concat_data_all_temperatures
from scipy.signal import argrelmax
import scipy.constants as cons
import scipy.special as sp
from scipy.interpolate import CubicSpline

#define which graphs to plot
plot_noise_spectrum = True
plot_corrected_noise_spectrum = True
plot_wavelength_spectrum = True
plot_wavelength_spectrum_zoom = True
plot_resonances = True

linearize = True
smooth = True
remove_overlap = False

resonance_folder = 'resonance_graphs/'

#gname_noise_spectrum = resonance_folder + 'noise_spectrum_all_t.pdf'
#gname_corrected_noise_spectrum = resonance_folder + 'corrected_noise_spectrum_all_t.pdf'
#gname_wavelength_spectrum = resonance_folder + 'wavelength_spectrum.pdf'
#gname_wavelength_spectrum_zoom = resonance_folder +'zoomed_wavelength.pdf'
#gname_resonances = resonance_folder + 'resonances.pdf'

gname_noise_spectrum = resonance_folder + 'noise_spectrum_all_t_smooth.pdf'
gname_corrected_noise_spectrum = resonance_folder + 'corrected_noise_spectrum_all_t_smooth.pdf'
gname_wavelength_spectrum = resonance_folder + 'wavelength_spectrum_smooth.pdf'
gname_wavelength_spectrum_zoom = resonance_folder +'zoomed_wavelength_smooth.pdf'
gname_resonances = resonance_folder + 'resonances_smooth.pdf'

ofile_resonant_lengths = 'resonant_lengths.txt'

relevant_wavelength = 4.2 

# Calculate cutoff frequency in MHz
c = cons.c
eps = cons.epsilon_0
mu = cons.mu_0
r = 5.0292e-3



def circ_cutoff_te(rad, m, n):
    f_cut = sp.jnp_zeros(m, n) / (2 * np.pi * np.sqrt(mu * eps) * rad)
    return f_cut  # in Hz


f_cut = circ_cutoff_te(r, 1, 1) / 1e6  # in MHz

# define temperatures of data taken
desired_temperatures = [40, 60, 70, 80, 90, 100, 110, 120]
#desired_temperatures = [40]
real_temperatures = [48, 58, 68, 71, 90, 100, 110, 120]
#real_temperatures = [48]
n_files = 31
#n_files = 1

# import and concatenate all data for all temperatures
data_dir = '2016_09_RFBackgroundMeasurements/'
template = data_dir + '2016_09_23_trace3_{}K_{}.dpt'

x_all_temp, y_all_temp, xfrag_all_temp, yfrag_all_temp, cf_all_temp = concat_data_all_temperatures(
    template, desired_temperatures, n_files, linearize, smooth, remove_overlap)

sig_rsa = 0.5  # dB
sig_rsa_lin = 10**(0.5 / 10)
sig_rsa_lin = 1e-4

# convert frequency to wavenumber (inverse wavelength)


def phase_velocity(f_cut, f):
    return c * (1 - f_cut**2 / f**2)**(-1 / 2)


def make_uniform_x(x, y):
    sample_spacing = x[1] - x[0]
    N_interpolation = int((x[-1] - x[1]) / sample_spacing)
    spl = CubicSpline(x, y)
    xs = np.linspace(x[0], x[-1], N_interpolation)
    ys = spl(xs)
    return xs, ys

# FFT the interpolation


def fft(x, y):
    sample_rate = 1 / (x[1] - x[0])
    n = y.size
    fft_x = np.fft.rfftfreq(n, 1. / sample_rate)
    fft_y = np.fft.rfft(y)
    psd = np.abs(fft_y)**2
    return fft_x[1:], psd[1:]


def find_resonances(fft_x, fft_y, max_lambda):
    ind_max = argrelmax(fft_y)
    ind_relevant = np.where(fft_x < max_lambda)
    ind_resonances = np.intersect1d(ind_max, ind_relevant)
    resonant_wavelengths = fft_x[ind_resonances]
    return resonant_wavelengths


def delete_multiple(index_compare, v):
    tolerance = 1e-15
    remainders = v % v[index_compare]
    ind_del = np.where((remainders <= tolerance) | (remainders <= tolerance))
    ind_del = np.setdiff1d(ind_del, index_compare)
    a = np.delete(v, ind_del)
    return a


def filter_resonances(resonant_v):
    if len(resonant_v) < 2:
        return resonant_v
    i = 0
    while True:
        resonant_v = delete_multiple(i, resonant_v)
        if ((i) == len(resonant_v) - 1):
            break  # plot spectrum vs wavenumber
        i += 1
    return resonant_v


wn_all_temp = {}
downsample_f = 24.2e3
wn_all_temp_s = {}
y_all_temp_s = {}
l_spectrum_x = {}
l_spectrum_y = {}
resonant_l = {}
resonance_uncertainty = {}
cavity_length = {}
cavity_length_uncertainty = {}

for destemp in desired_temperatures:
    x = x_all_temp[destemp] + downsample_f
    y = y_all_temp[destemp]
    wn_all_temp[destemp] = x * 1e6 / phase_velocity(f_cut, x)
    x = wn_all_temp[destemp]
    xs, ys = make_uniform_x(x, y)
    ys -= np.mean(ys)
    wn_all_temp_s[destemp] = xs
    y_all_temp_s[destemp] = ys
    fft_xs, fft_ys = fft(xs, ys)
    l_spectrum_x[destemp] = fft_xs
    l_spectrum_y[destemp] = fft_ys
    resonances = find_resonances(fft_xs, fft_ys, relevant_wavelength)
    resonant_l[destemp] = resonances
    filtered_resonances = filter_resonances(resonances)
    resonance_uncertainty[destemp] = np.mean(np.diff(fft_xs) / 2)
    cavity_length[destemp] = filtered_resonances / 2
    cavity_length_uncertainty[destemp] = resonance_uncertainty[destemp] / 2

# print(cavity_length)
f = open(ofile_resonant_lengths, 'w')
f.write(repr(cavity_length))
f.write('\n')
f.write(repr(cavity_length_uncertainty))
f.close()

# plot RF noise background
if plot_noise_spectrum:
    for destemp, realtemp in zip(desired_temperatures, real_temperatures):
        x = x_all_temp[destemp]
        y = y_all_temp[destemp]
#        sig_y = sig_y_all_temp[destemp]
#        print(np.mean(sig_y))
    #    y -= np.mean(y)
        plt.plot(x, y, '.', ms=1, label=str(realtemp) + ' K', rasterized = True)
#        plt.errorbar(x, y, sig_y,  label = str(realtemp) + ' K')

    legend = plt.legend(loc='best', markerscale=10)

    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Noise Spectrum [nW/Hz]')
    plt.tight_layout()
    plt.savefig(gname_noise_spectrum)
    plt.close()

# plot spectrum vs wavenumber
if plot_corrected_noise_spectrum:
    for destemp, realtemp in zip(desired_temperatures, real_temperatures):
        x = wn_all_temp[destemp]
        y = y_all_temp[destemp]
        plt.plot(x, y, '.', ms=1, label=str(realtemp) + ' K', rasterized = True)

    legend = plt.legend(loc='best', markerscale=10)

    plt.xlabel(r'Wavenumber $\~{\nu}$ [1/m]')
    plt.ylabel('Noise Spectrum [nW/Hz]')
    plt.tight_layout()
    plt.savefig(gname_corrected_noise_spectrum)
    plt.close()

# plot spectrum vs wavenumber
if plot_wavelength_spectrum:
    for destemp, realtemp in zip(desired_temperatures, real_temperatures):
        x = l_spectrum_x[destemp]
        y = l_spectrum_y[destemp]
        plt.plot(x, y, label=str(realtemp) + ' K', rasterized = True)

    legend = plt.legend(loc='best', markerscale=10)

    plt.xlabel(r'Wavelength $\lambda$ [m]')
    plt.ylabel('Power')
    plt.axvline(relevant_wavelength, lw = 1)
    plt.loglog()
    plt.tight_layout()
    plt.savefig(gname_wavelength_spectrum)
    plt.close()

# relevant cavity lengths

# plot spectrum vs wavenumber
if plot_wavelength_spectrum_zoom:
    for destemp, realtemp in zip(desired_temperatures, real_temperatures):
        x = l_spectrum_x[destemp]
        y = l_spectrum_y[destemp]
        ind_zoom = np.where(x < relevant_wavelength)
        x = x[ind_zoom]
        y = y[ind_zoom]
        plt.plot(x, y, label=str(realtemp) + ' K')

    legend = plt.legend(loc='best', markerscale=10)

    plt.xlabel(r'Wavelength $\lambda$ [m]')
    plt.ylabel('Power')
    plt.loglog()
    plt.tight_layout()
    plt.savefig(gname_wavelength_spectrum_zoom)
    plt.close()

resonator_lengths = np.array(
    [18.759, 145.89 - 18.759, 150 - 23.839, 145.89]) * 1e-2
#resonator_labels = ['short -> beginning of waveguide adaptor', 'beginning of waveguide adaptor -> beginning of amplifier adaptor', 'end of waveguide adaptor -> end of amplifier adaptor', 'short -> beginning of amplifier adaptor']
resonator_colors = ['red', 'blue', 'green', 'cyan']
if plot_resonances:
    for destemp, realtemp in zip(desired_temperatures, real_temperatures):
        y = cavity_length[destemp]
        sig_y = cavity_length_uncertainty[destemp]
        x = realtemp * np.ones_like(y)
        plt.errorbar(x, y, yerr=sig_y, fmt='o')

    for length, res_color in zip(resonator_lengths,
                                 resonator_colors):
        plt.axhline(length, lw=1, color=res_color)
    plt.ylabel(r'Resonant Lengths $\lambda$/2 [m]')
    plt.xlabel('Temperature [K]')
    plt.tight_layout()
    plt.savefig(gname_resonances)
    plt.close()
