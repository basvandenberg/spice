import numpy
from matplotlib import pyplot
from spica.plotpy import color

class Scatter(object):

    def __init__(self, data, labels, xlabel, ylabel, legend,
                 colors=None, min_val=-3.0, max_val=3.0):
        '''
        data: n x 2 numpy array (n is number of objects)
        labels: n x 1 list with int labels (labels are 0, 1, 2, ...)
        xlabel: feature name for the x-axis
        ylabel: feature name for the y-axis
        legend: label legend
        colors: mapping label to color
        '''
        self.data = data
        self.labels = labels
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.min_val = min_val
        self.max_val = max_val
        if colors:
            self.colors = colors
        else:
            self.colors = color.color_dict()

    def get_norm_svg_scatter(self, title='', svg_id=''):

        '''
        SVG scatter plot.
        Intended for normalized data (mean 0.0, std 1.0)
        '''

        x_label = title

        # default plot settings
        width = 550
        height = 550

        radius = 4

        margin = 30
        title_x = margin + 5
        title_y = margin - 5
        tick_height = 4
        tick_width = 4
        
        # derived
        x_orig = margin
        y_orig = height - margin
        x_max = width - margin
        y_max = margin
        y_range = y_orig - y_max
        x_range = x_max - x_orig

        # map feature value to xposition
        def to_y(feat_val):
            result = y_orig - int(round( float(y_range) * 
                    ((feat_val - self.min_val) / (self.max_val - self.min_val))))
            return result

        # map feature value to xposition
        def to_x(feat_val):
            result = x_orig + int(round( float(x_range) *
                    ((feat_val - self.min_val) / (self.max_val - self.min_val))))
            return result

        # first the svg open tag
        result = '''<svg class="svg_scatter" id="%s" xmlns="http://www.w3.org/2000/svg" version="1.1" width="%i" height="%i">''' % (svg_id, width, height)

        # define style
        class_colors = []
        for lab in set(self.labels):
            class_colors.append(color.int_str(self.colors[lab]))

        axis_line_style = "stroke:#888a85;stroke-width:1px;"
        axis_text_style = "fill:#555753;font-size:10px"
        grid_style = "stroke:#d3d7cf;stroke-width:1px;"
        title_style = "fill:#2e3436;font-size:13px;"
        legend_style = "stroke-width:0.5;stroke: #2e3436;"
        
        # add the title (change coordinates here if needed)
        result += '''
    <text style="%s" x="%i" y="%i">%s</text>''' % (title_style, title_x, title_y, title)

        ########################################################################
        # Axes
        ########################################################################
        
        # draw xaxis
        result += '''    
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (axis_line_style, x_orig, x_max, y_orig, y_orig)
        
        # calculate positions and labels
        num_xticks = 11
        xstep = (self.max_val - self.min_val) / (num_xticks - 1)
        xvals = numpy.arange(self.min_val, self.max_val + 0.1*xstep, xstep)
        xlabels = ['%.1f' % (x) for x in xvals]
        xpositions = [to_x(x) for x in xvals]
        #print '\nx-axis'
        #print xpositions
        #print xlabels

        # draw ticks and labels
        for index in xrange(num_xticks):
            xpos = xpositions[index]
            xval = xvals[index]
            xlab = xlabels[index]
            result += '''    
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (axis_line_style, xpos, xpos, y_orig, y_orig + tick_height)
            if not(xval == self.min_val or xval == self.max_val): 
                result += '''
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (grid_style, xpos, xpos,  y_orig, y_max)
            shift = 10 if xval < 0.0 else 7
            result += '''
    <text style="%s" x="%i" y="%i">%s</text>''' % (axis_text_style, xpos - shift, y_orig + 12, xlab)
        
        # draw xaxis2
        result += '''    
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (axis_line_style, x_orig, x_max, y_max, y_max)
        
        # draw yaxis
        result += '''
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (axis_line_style, x_orig, x_orig, y_orig, y_max)
        
        # calculate y tick-positions and ylabels
        num_yticks = 11
        ystep = (self.max_val - self.min_val) / (num_xticks - 1)
        yvals = numpy.arange(self.min_val, self.max_val + 0.1*ystep, ystep)
        ylabels = ['%.1f' % (y) for y in yvals]
        ypositions = [to_y(y) for y in yvals]
        #print '\ny-axis'
        #print ypositions
        #print ylabels

        # draw ticks and labels
        for index in xrange(num_yticks):
            ypos = ypositions[index]
            yval = yvals[index]
            ylab = ylabels[index]
            result += '''    
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (axis_line_style, x_orig - tick_width, x_orig, ypos, ypos)
            if not(yval == self.min_val or yval == self.max_val): 
                result += '''
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (grid_style, x_orig, x_max, ypos, ypos)
            shift = 37
            if(yval < 1000):
                shift = 32
            if(yval < 100):
                shift = 27
            if(yval < 10):
                shift = 22
            result += '''
    <text style="%s" x="%i" y="%i">%s</text>''' % (axis_text_style, x_orig - shift, ypos + 3, ylab)

        # draw yaxis2
        result += '''
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (axis_line_style, x_max, x_max, y_orig, y_max)

        # draw the scatter plot
        for rowi, (x, y) in enumerate(self.data):
                x = to_x(x)
                y = to_y(y)
                #print '%i\t%i\t%i\t%i' % (bi, x, current_data[bi], h)
                c = color.int_str(self.colors[self.labels[rowi]])
                result += '''        
    <circle style="fill:rgb%s;fill-opacity:0.3;stroke:rgb%s;stroke-width:1;" cx="%i" cy="%i" r="%i"/>''' % (c, c, x, y, radius)
    
        # draw yaxis1 legend
        ypos = y_max + 10
        for index, leg in enumerate(self.legend):
            c = color.int_str(self.colors[index])
            result += '''
    <rect x="%i" y="%i" width="%i" height="%i" style="%s;fill:rgb%s;"/>''' % (x_orig + 10, ypos, 10, 10, legend_style, c)
            result += '''
    <text class="axis" x="%i" y="%i">%s</text>''' % (x_orig + 25, ypos + 8, leg)
            ypos += 15

        # and finally the closing tag and it's 22:00, time to close the laptop.
        result += '''
</svg>
'''
        return result 
