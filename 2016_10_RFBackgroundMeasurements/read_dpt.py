import xmltodict
import numpy as np

def read_rsa_dpt(file):
    xml_obj = xmltodict.parse(open(file).read())
    data_xml = xml_obj['RSAPersist']['Internal']['Composite']['Items']['Composite']['Items']['Waveform']
    y = data_xml['y']
    y = np.array(y)
    y = y.astype(np.float)
    xmin = int(data_xml['XStart'])
    xmax = int(data_xml['XStop'])
    npoints = data_xml['Count']
    x = np.linspace(xmin, xmax, npoints)/1e6
    x_unit = data_xml['XUnits']
    y_unit = data_xml['YUnits']
    return x,y


