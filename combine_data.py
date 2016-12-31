import numpy as np
import read_dpt
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def concat_data_one_temp(
        template_w_t,
        n_files,
        linearize=False,
        smooth=False,
        remove_overlap=False):
    files = []
    x = [0] * (n_files)
    y = [0] * (n_files)
    cf = [0] * (n_files)
    filter_sigma = 10

    for i in range(0, n_files):
        files.append(template_w_t.format(i))
    def dBm_to_mW(p):
        return 10**(p / 10)
    for i in range(n_files):
        x[i], y[i], cf[i] = read_dpt.read_rsa_dpt(files[i])
        if remove_overlap:
            x[i] = x[i][236:565]
            y[i] = y[i][236:565]
        if linearize:
            y[i] = dBm_to_mW(y[i]) * 1e6  # nW
    x_total = np.concatenate(x)
    y_total = np.concatenate(y)
    ind_srt = np.argsort(x_total)
    x_total = x_total[ind_srt]
    y_total = y_total[ind_srt]
    if smooth:
        y_total = gaussian_filter(y_total, filter_sigma)
    return(x_total, y_total, x, y, cf)


def concat_data_all_temperatures(
        template_wo_t,
        temperatures,
        n_files,
        linearize=False,
        smooth=False,
        remove_overlap=False):
    x_all_total = {}
    y_all_total = {}
    sig_y_all_total = {}
    x_total = {}
    y_total = {}
    sig_y_total = {}
    cf_total = {}

    for destemp in temperatures:
        template_temp = template_wo_t.format(destemp, {})
        x_all_total[destemp], y_all_total[destemp], x_total[destemp], y_total[destemp], cf_total[destemp] = concat_data_one_temp(
                template_temp, n_files, linearize, smooth, remove_overlap)
    return x_all_total, y_all_total, x_total, y_total, cf_total
