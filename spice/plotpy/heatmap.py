'''
Created on Fri Oct 1, 2010

@author: Bastiaan van den Berg
'''

import numpy
#from scipy.cluster import hierarchy
from matplotlib import pyplot, colors, cm, gridspec, colorbar, ticker

from spica.plotpy import color

def heatmap_fig(data, xlab, ylab, file_name, vmin=-3.0, vmax=3.0):
    
    fig = pyplot.figure(figsize=(8,8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    
    ax0 = pyplot.subplot(gs[0, 0])
    _heatmap_axes(ax0, data, xlab, ylab, vmin, vmax)
    
    ax1 = pyplot.subplot(gs[0, 1])
    #colorbar.ColorbarBase(ax1, cmap=my_cmap())
    colorbar.ColorbarBase(ax1, cmap=cm.get_cmap('RdBu'))
    step = (vmax - vmin) / 10.0
    ticklab = list(numpy.arange(vmin, vmax, step))
    ticklab.append(vmax)
    ticklab = [round(t, 1) for t in ticklab]
    ax1.yaxis.set_ticklabels(ticklab)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(9)
    
    fig.savefig(file_name, bbox_inches='tight')


def heatmap_labeled_fig(data, xlab, ylab, label_lists, class_names, file_path, 
                        vmin=-3.0, vmax=3.0):
    '''
    returns figure with heatmap of provided data, and a column for each
    provided label list.
    '''

    (nrows, ncols) = data.shape    

    width = min(6.4, ncols * 0.5)
    height = nrows * 0.03
    fig = pyplot.figure(figsize=(width, height))

    pyplot.subplots_adjust(bottom=0.2)

    num_rows = 1
    num_cols = 2 + len(label_lists)

    ratios = [1] * num_cols
    ratios[0] = data.shape[1]

    gs = gridspec.GridSpec(1, num_cols, width_ratios=ratios)
    gs.update(left=0.01, wspace=0.05)

    labeling_ax = []
    for i in range(len(label_lists)):
        axi = pyplot.subplot(gs[0, i + 1])
        _labeling_axes(axi, label_lists[i])
        labeling_ax.append(axi)
    
    ax1 = pyplot.subplot(gs[0, -1])
    colorbar.ColorbarBase(ax1, cmap=my_cmap())
    step = (vmax - vmin) / 10.0
    ticklab = list(numpy.arange(vmin, vmax, step))
    ticklab.append(vmax)
    ticklab = [round(t, 1) for t in ticklab]
    ax1.yaxis.set_ticklabels(ticklab)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(9)

    ax0 = pyplot.subplot(gs[0, 0])
    _heatmap_axes(ax0, data, xlab, ylab, vmin, vmax)
    
    # draw legend
    '''
    colormap = my_cmap_2lab()
    for cname_list in class_names:
        nlab = len(cname_list)
        for i, cname in enumerate(cname_list):
            w = 0.2
            h = 3
            #x = 0
            #y = 33.5 * height + i * (h + 1)
            x = num_cols * 1.55
            y = (-2 - (nlab * (h + 2))) + i * (h + 2)
            r = pyplot.Rectangle((x, y), w, h, facecolor=colormap(i))
            r.set_clip_on(False)
            x = 1.15 * ((ncols - 2) / float(ncols))
            tmph = fig.get_figheight()
            y = 1.0 - (0.5 / tmph)
            print
            print tmph
            print 0.5 / tmph
            print y
            print
            fig.text(x, y, cname, family='sans-serif', size=8)
            ax0.add_patch(r)
    '''

    fig.savefig(file_path + '.png', bbox_inches='tight')
    fig.savefig(file_path + '.svg', bbox_inches='tight')
    #fig.savefig(file_path + '.png')
    #fig.savefig(file_path + '.svg')

def _heatmap_axes(ax, data, xlab, ylab, vmin, vmax):
    
    # check sizes...
    (numy, numx) = data.shape
    if not(numx == len(xlab)):
        print('Error: incorrect number of x-labels')
        return
    if not(numy == len(ylab)):
        print('Error: incorrect number of y-labels')

    # set x-labels, feature names, at the top
    ax.xaxis.set_ticks(range(numx))
    ax.xaxis.set_ticklabels(xlab)
    ax.xaxis.set_ticks_position('top')

    # rotate x-axis tick labels
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(90)
        label.set_fontsize(8)
    
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    
    # remove y-labels
    ylabels = []
    
    # change size y-axis tick labels
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(4)

    # remove all tick markers
    for t in ax.yaxis.get_ticklines():
        t.set_markersize(0)
    for t in ax.xaxis.get_ticklines():
        t.set_markersize(0)

    # plot heatmap
    hm = ax.imshow(data, origin='upper', 
                         extent=None, 
                         aspect='auto',
                         vmin = vmin,
                         vmax = vmax)
    hm.set_interpolation('nearest')
    hm.set_cmap(my_cmap())
    #hm.set_cmap(cm.get_cmap('RdBu'))

def _labeling_axes(ax, label_list):
    
    # label_list must be one column and several rows

    colormap = my_cmap_2lab()
    lset = set(label_list)
    if(len(lset) > 2):
        colormap = my_cmap_mlab(len(lset))

    # add extra column (copy) so that we have a 2d array and can use
    # imshow 
    im_list = numpy.array([label_list, label_list]).transpose()
    
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    lm = ax.imshow(im_list, origin='upper',
                               extent=None,
                               aspect='auto',
                               vmin = 0,
                               vmax = len(lset))
    lm.set_interpolation('nearest')
    lm.set_cmap(colormap)

def my_cmap():
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.5, 0.463, 0.463),
                     (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.5, 0.714, 0.714),
                       (1.0, 1.0, 1.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.5, 0.929, 0.929),
                      (1.0, 1.0, 1.0))}
    return colors.LinearSegmentedColormap('my_cmap', cdict, 256)

def my_cmap_2lab():
    color_list = ['#888a85', '#eeeeec']
    return colors.ListedColormap(color_list, name='my_cmap_2lab', N=2)

def my_cmap_mlab(n):
    return cmap_discretize(cm.jet, n)
    #return cm.spectral
    #color_list = ['#000000', '#f57900', '#c17d11', '#73d216', '#3465a4',
    #              '#75507b', '#cc0000', '#ff0000', '#00ff00', '#0000ff',
    #              '#ffff00', '#00ffff', '#ff00ff', '#550000', '#005500', 
    #              '#000055', '#555500', '#005555', '#550055', '#0055ff',
    #              '#ffffff', '#edd400']
    #return colors.ListedColormap(color_list, name='my_cmap_mlab', N=256)

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
        cmap = cm.get_cmap(cmap)
    #colors_i = numpy.concatenate((numpy.linspace(0, 1., N), (0.,0.,0.,0.)))
    #colors_rgba = cmap(colors_i)
    colors_rgba = color.colors[:N]
    #indices = numpy.linspace(0, 1., N+1)
    #indices = range(N+1)
    #cdict = {}
    #for ki,key in enumerate(('red','green','blue')):
    #    cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in xrange(N+1) ]
    # Return colormap object.
    #return colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)
    return colors.ListedColormap(colors_rgba, N)
