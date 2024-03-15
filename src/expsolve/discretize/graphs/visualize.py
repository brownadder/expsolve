import numpy as np
import matplotlib as mpl
from matplotlib.patches import RegularPolygon, Circle
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
import torch




def drawactivations(ax, h, centers, intensities, cmap=mpl.colormaps['Reds'], 
                    activations=None, node = lambda x,y,h: Circle(xy=(x, y), radius=h/4)):
    
    if activations is None:
        nodes =[node(x,y,h) for (x,y) in centers]
        activations = PatchCollection(nodes, cmap=cmap)
        activations.set_array(intensities)
        ax.add_collection(activations)
        return activations
    else:
        activations.set_array(intensities)
        return activations
    
def drawhoneycomb(ax, xr, h, centers, intensities=None, facecolor=mcolors.XKCD_COLORS['xkcd:black'], 
                  edgecolor=mcolors.XKCD_COLORS['xkcd:dark grey blue'], activationscolormap=mpl.colormaps['Reds'], 
                  activations=None, node = lambda x,y,h: Circle(xy=(x, y), radius=h/4)):
    
    hexagons =[RegularPolygon(center, numVertices=6, radius=h/np.sqrt(3), orientation=torch.pi/6) for center in centers]
    hexagon_collection = PatchCollection(hexagons, color=facecolor, edgecolor=edgecolor)
    ax.add_collection(hexagon_collection)

    if intensities is not None:
        drawactivations(ax, h, centers, intensities, cmap=activationscolormap, activations=activations, node=node)

    ax.set_xlim(xr[0][0], xr[0][1])
    ax.set_ylim(xr[1][0], xr[1][1])
    ax.set_aspect('equal', 'box')
