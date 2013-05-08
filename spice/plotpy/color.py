import numpy
import matplotlib
from matplotlib import pyplot

colors = [(0.90, 0.60, 0.00, 1.00),
          (0.35, 0.70, 0.90, 1.00),
          (0.00, 0.60, 0.50, 1.00),
          (0.95, 0.90, 0.25, 1.00),
          (0.00, 0.45, 0.70, 1.00),
          (0.80, 0.40, 0.00, 1.00),
          (0.80, 0.60, 0.70, 1.00),
          (0.00, 0.00, 0.00, 1.00),
        
          (0.90, 0.60, 0.00, 0.50),
          (0.35, 0.70, 0.90, 0.50),
          (0.00, 0.60, 0.50, 0.50),
          (0.95, 0.90, 0.25, 0.50),
          (0.00, 0.45, 0.70, 0.50),
          (0.80, 0.40, 0.00, 0.50),
          (0.80, 0.60, 0.70, 0.50),
          (0.00, 0.00, 0.00, 0.50)]

def color_dict():
    '''
    Color blind-proof color dictionary.
    '''
    return dict(enumerate(colors))

def percentage_str(c):
    result = '('
    for i in xrange(3):
        result += str(int(c[i] * 100)) + '%,'
    result += str(c[3]) + ')'
    return result

def int_str(c):
    result = '('
    for i in xrange(3):
        result += str(int(c[i] * 255)) + ','
    result = result[:-1] + ')'
    return result

def d_cmap():
    return cmap_discretize('Set1', 9)

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = pyplot.get_cmap(cmap)
    colors_i = numpy.concatenate((numpy.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = numpy.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)
