from .files import localfilepath
from numpy import genfromtxt
from matplotlib.colors import ListedColormap

def orangetealmap():
    filepath = localfilepath('ot.csv', __file__)
    ot = genfromtxt(filepath, delimiter=',')
    otmap = ListedColormap(ot, name='OrangeTeal')
    return otmap
