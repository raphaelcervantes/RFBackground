import numpy as np
import read_dpt

def combine_data(template, n_files):
    files = []
    x = [0]*n_files
    y = [0]*n_files

    for i in range(1,n_files+1):
        files.append(file_template.format(i))
    
    for i in range(n_files):
        x[i],y[i] = read_dpt.read_rsa_dpt(files[i])

    x_total = np.concatenate(x)
    y_total = np.concatenate(y)


def bin_data(x,y, xbins = 801):
    n, bin_edges = np.histogram(x, bins=xbins)
    sy, bin_edges = np.histogram(x, bins=xbins, weights=y)
    sy2, bin_edges = np.histogram(x, bins=xbins, weights=y*y)

    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    x_bin = bin_edges
    print(len(x_bin))
    print(len(mean))
    print(len(x))
    print(len(y))
#    print(x_bin)
#    print(x)
#    print(x_bin==x)
    return  x_bin, mean
