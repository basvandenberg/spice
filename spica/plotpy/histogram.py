import numpy
from matplotlib import pyplot
from ssbc.plotpy import color

def rounded_maxy(maxy):
    boundaries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 
                  10, 20, 30, 40, 50, 60, 70, 80, 90, 
                  100, 200, 300, 400, 500, 600, 700, 800, 900, 
                  1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 
                  10000, 20000, 30000, 40000, 50000, 
                  60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 1000000]
    bound_i = 0
    while(maxy > boundaries[bound_i] and bound_i < len(boundaries)):
        bound_i += 1
    return boundaries[bound_i]

#
# TODO: histogram calculation is slow... improve this
#

class Histogram(object):

    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.colors = color.color_dict()

    def settings(self, min_val=-10.0, max_val=10.0, num_bins=10, legend='', 
                 color='#555753', maxy=None):
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
        self.step_size = (max_val - min_val) / num_bins
        self.legend = legend
        self.color = color
        self.maxy = maxy

    def calc(self):
        self.hist_data = numpy.histogram(self.raw_data, self.num_bins, range=(self.min_val, self.max_val))[0]

    def calc_slow(self): 

        for val in self.raw_data:
    
            self.hist_data = [0] * self.num_bins
            self.cnt_bfr = 0
            self.cnt_aft = 0
            
            for val in self.raw_data:
                
                current_bin = -1
                
                current_val = self.min_val
                while(val > current_val and current_bin < self.num_bins):
                    current_bin += 1
                    current_val += self.step_size
                    
                if(current_bin < 0):
                    self.cnt_bfr += 1
                elif(current_bin < self.num_bins):
                    self.hist_data[current_bin] += 1
                else:
                    self.cnt_aft += 1

    def get_norm_svg_hist(self, others=[], title='', svg_id=''):

        '''
        SVG histogram.
        Intended for normalized data (mean 0.0, std 1.0)
        '''

        x_label = title

        # default plot settings
        width = 640 
        height = 210

        margin = 30
        title_x = margin + 5
        title_y = margin - 5
        tick_height = 4
        tick_width = 4
        bar_margin = 3
        inner_bar_margin = 1
        
        # derived
        num_hists = 1 + len(others)
        x_orig = margin
        y_orig = height - margin
        x_max = width - margin
        y_max = margin
        y_range = y_orig - y_max
        x_range = x_max - x_orig
        binwidth = int((x_max - x_orig) / self.num_bins)
        bar_width = ((binwidth - 2 * bar_margin) / num_hists) - inner_bar_margin

        # map bincount to yposition
        def map_count2y(count):
            return y_orig - int(round((float(y_range) / self.maxy) * count))

        # map (norm) feature value to xposition
        def map_val2x(val):
            return x_orig + int(round((float(x_range) / (self.max_val - self.min_val)) * (val - self.min_val)))

        # obtain all hist data
        all_hists = [self]
        all_hists.extend(others)

        # first the svg style sheet
        result = '''<svg class="svg_hist" id="%s" xmlns="http://www.w3.org/2000/svg" version="1.1" width="%i" height="%i">''' % (svg_id, width, height)

        # define style
        bar_colors = []
        for hist_i in xrange(len(all_hists)):
            bar_colors.append(color.int_str(all_hists[hist_i].color))
        axis_line_style = "stroke:#888a85;stroke-width:1px;"
        axis_text_style = "fill:#555753;font-size:10px"
        grid_style = "stroke:#d3d7cf;stroke-width:1px;"
        title_style = "fill:#2e3436;font-size:13px;"
        legend_style = "stroke-width:0.5;stroke: #2e3436;"
        
        stub = '''
        rect.legend { 
          stroke-width: 0.5;
          stroke: #2e3436;
        }
        rect#pointer {
          fill: #cc0000;
        }
        line#pred {
          stroke: #3465a4;
          stroke-width: 1px;
        }
        text#pred {
          fill: #3465a4;
        }
'''
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
        num_xticks = self.num_bins + 1
        xstep = (self.max_val - self.min_val) / self.num_bins
        xvals = numpy.arange(self.min_val, self.max_val + 0.1*xstep, xstep)
        xlabels = ['%.1f' % (x) for x in xvals]
        xpositions = [map_val2x(x) for x in xvals]
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
        ystep = float(self.maxy) / (num_yticks - 1)
        yvals = numpy.arange(0, self.maxy + 0.1*ystep, ystep)
        ylabels = ['%i' % (y) for y in yvals]
        ypositions = [map_count2y(y) for y in yvals]
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
            if not(yval == 0 or yval == self.maxy): 
                result += '''
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (grid_style, x_orig, x_max, ypos, ypos)
            shift = 27
            if(yval < 1000):
                shift = 22
            if(yval < 100):
                shift = 17
            if(yval < 10):
                shift = 12
            result += '''
    <text style="%s" x="%i" y="%i">%s</text>''' % (axis_text_style, x_orig - shift, ypos + 3, ylab)

        # draw yaxis2
        result += '''
    <line style="%s" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (axis_line_style, x_max, x_max, y_orig, y_max)

        # draw the histogram   
        for index in xrange(len(all_hists)):

            current_data = all_hists[index].hist_data

            # use the number of bins of this object...
            #print '\ndata: (bin index, xval, count, height)'
            for bi in xrange(self.num_bins):
                y = map_count2y(current_data[bi])
                x = x_orig + bar_margin +  bi * binwidth + index * (bar_width + inner_bar_margin)
                h = y_orig - y
                #print '%i\t%i\t%i\t%i' % (bi, x, current_data[bi], h)
                if(h > 0):
                    result += '''        
    <rect style="fill:rgb%s;" x="%i" y="%i" width="%i" height="%i"/>''' % (bar_colors[index], x, y, bar_width, h)
    
        # draw yaxis1 legend
        ypos = y_max + 10
        for index, h in enumerate(all_hists):
            l = h.legend
            result += '''
    <rect x="%i" y="%i" width="%i" height="%i" style="%s;fill:rgb%s;"/>''' % (x_orig + 10, ypos, 10, 10, legend_style, bar_colors[index])
            result += '''
    <text class="axis" x="%i" y="%i">%s</text>''' % (x_orig + 25, ypos + 8, l)
            ypos += 15

        # and finally the closing tag and it's 22:00, time to close the laptop.
        result += '''
</svg>
'''
        return result 
























    def get_norm_svg_hist_bu(self, others=[], title='', svg_id=''):

        '''
        SVG histogram.
        Intended for normalized data (mean 0.0, std 1.0)
        '''

        x_label = title

        # default plot settings
        width = 640 
        height = 140

        margin = 20
        title_x = 40
        title_y = 10
        tick_height = 4
        tick_width = 4
        bar_margin = 3
        inner_bar_margin = 1
        
        # derived
        num_hists = 1 + len(others)
        x_orig = margin
        y_orig = height - margin
        x_max = width - margin
        y_max = margin
        y_range = y_orig - y_max
        x_range = x_max - x_orig
        binwidth = int((x_max - x_orig) / self.num_bins)
        bar_width = ((binwidth - 2 * bar_margin) / num_hists) - inner_bar_margin

        # map bincount to yposition
        def map_count2y(count):
            return y_orig - int(round((float(y_range) / self.maxy) * count))

        # map (norm) feature value to xposition
        def map_val2x(val):
            return x_orig + int(round((float(x_range) / (self.max_val - self.min_val)) * (val - self.min_val)))

        # obtain all hist data
        all_hists = [self]
        all_hists.extend(others)

        # first the svg style sheet
        result = '''
<svg class="svg_hist" id="%s" xmlns="http://www.w3.org/2000/svg" version="1.1" width="%i" height="%i">''' % (svg_id, width, height)
        result += '''
    <style type="text/css" >
    <![CDATA[
'''
        for hist_i in xrange(len(all_hists)):
            result += '''
        rect.bar%i {
          fill: rgba%s;
            }''' % (hist_i, color.percentage_str(all_hists[hist_i].color))

        result += '''
        rect.legend { 
          stroke-width: 0.5;
          stroke: #2e3436;
        }
        rect#pointer {
          fill: #cc0000;
        }
        line.axis {
          stroke: #888a85;
          stroke-width: 1px;
        }
        line.grid {
          stroke: #d3d7cf;
          stroke-width: 1px;
        }
        text.axis {
          fill: #555753;
          font-size: 10px;
        }
        line#pred {
          stroke: #3465a4;
          stroke-width: 1px;
        }
        text#pred {
          fill: #3465a4;
        }
        text.title {
          fill: #2e3436;
          font-size: 13px;
        }
    ]]>
    </style>
    '''

        # add the title (change coordinates here if needed)
        result += '''
    <text class="title" x="%i" y="%i">%s</text>''' % (title_x, title_y, title)

        ########################################################################
        # Axes
        ########################################################################
        
        # draw xaxis
        result += '''    
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig, x_max, y_orig, y_orig)
        
        # NEW STYLE nice :) xticks
        # calculate positions and labels
        num_xticks = self.num_bins + 1
        xstep = (self.max_val - self.min_val) / self.num_bins
        xvals = numpy.arange(self.min_val, self.max_val + 0.1*xstep, xstep)
        xlabels = ['%.1f' % (x) for x in xvals]
        xpositions = [map_val2x(x) for x in xvals]
        #print '\nx-axis'
        #print xpositions
        #print xlabels

        # draw ticks and labels
        for index in xrange(num_xticks):
            xpos = xpositions[index]
            xval = xvals[index]
            xlab = xlabels[index]
            result += '''    
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (xpos, xpos, y_orig, y_orig + tick_height)
            shift = 10 if xval < 0.0 else 7
            result += '''
    <text class="axis" x="%i" y="%i">%s</text>''' % (xpos - shift, y_orig + 12, xlab)
        
        # draw xlabel
        #result += '''
#    <text class="axis" x="%i" y="%i">%s</text>''' % ((width / 2) - 35, y_orig + 24, x_label)    

        # draw xaxis2
        result += '''    
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig, x_max, y_max, y_max)
        
        # draw yaxis
        result += '''
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig, x_orig, y_orig, y_max)
        
        # NEW STYLE nice :) yticks
        # calculate y tick-positions and ylabels
        num_yticks = 11
        ystep = float(self.maxy) / (num_yticks - 1)
        yvals = numpy.arange(0, self.maxy + 0.1*ystep, ystep)
        ylabels = ['%i' % (y) for y in yvals]
        ypositions = [map_count2y(y) for y in yvals]
        #print '\ny-axis'
        #print ypositions
        #print ylabels

        # draw ticks and labels
        for index in xrange(num_yticks):
            ypos = ypositions[index]
            yval = yvals[index]
            ylab = ylabels[index]
            result += '''    
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig - tick_width, x_orig, ypos, ypos)
            if not(yval == 0 or yval == self.maxy): 
                result += '''
    <line class="grid" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig, x_max, ypos, ypos)
            shift = 25
            if(yval < 100):
                shift = 18
            if(yval < 10):
                shift = 12
            result += '''
    <text class="axis" x="%i" y="%i">%s</text>''' % (x_orig - shift, ypos + 3, ylab)

        # draw ylabel1
        labx = x_orig - 32
        laby = (height / 2) + 25
        #result += '''
#    <text class="axis" x="%i" y="%i" transform="rotate(-90 %i, %i)">count</text>''' % (labx, laby, labx, laby)    

        # draw yaxis1 legend
        #ypos = y_max + 10
        #for index in xrange(len(pred_classes)):
        #    pc = pred_classes[index]
        #    leg = legend_strings[index]
        #    result += '''
        #    <rect x="%i" y="%i" width="%i" height="%i" class="%s legend"/>''' % (x_orig + 10, ypos, 10, 10, pc)
        #    result += '''
        #    <text class="axis" x="%i" y="%i">%s</text>''' % (x_orig + 25, ypos + 8, leg)
        #    ypos += 15

        # draw yaxis2
        result += '''
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_max, x_max, y_orig, y_max)

        # draw the histogram   
        for index in xrange(len(all_hists)):

            current_data = all_hists[index].hist_data

            # use the number of bins of this object...
            #print '\ndata: (bin index, xval, count, height)'
            for bi in xrange(self.num_bins):
                y = map_count2y(current_data[bi])
                x = x_orig + bar_margin +  bi * binwidth + index * (bar_width + inner_bar_margin)
                h = y_orig - y
                #print '%i\t%i\t%i\t%i' % (bi, x, current_data[bi], h)
                if(h > 0):
                    result += '''        
    <rect class="bar%i" x="%i" y="%i" width="%i" height="%i"/>''' % (index, x, y, bar_width, h)
    
        # and finally the closing tag and it's 22:00, time to close the laptop.
        result += '''
</svg>
'''
        return result 

















    def get_mpl_hist(self, others=[], title=''):
        
        # obtain all hist data
        all_hist_data = [self.hist_data]
        leg = [self.legend]
        for h in others:
            all_hist_data.append(h.hist_data)
            leg.append(h.legend)
        
        for h in others:
            leg.append

        # x-axis indices
        ind = numpy.arange(self.num_bins)

        # the width of the bars
        margin = 0.1
        width = (1.0 - 2 * margin) / len(all_hist_data)

        # create matplotlib figure and axes
        fig = pyplot.figure()
        ax = fig.add_subplot(111)

        # iterate over data
        x_offset = 0.0
        rects = []
        for index in xrange(len(all_hist_data)):

            hdata = all_hist_data[index]
            rects.append(ax.bar(ind + (margin + x_offset), hdata, width, color=self.colors[index]))
            x_offset += width

        # x-labels
        xlabs = numpy.arange(self.min_val, self.max_val + 0.5 * self.step_size, self.step_size)
        xlabs = ['%.1f' % (v) for v in xlabs]

        # add some
        ax.set_xlim((0.0, self.num_bins))
        if(self.maxy):
            ax.set_ylim((0.0, self.maxy))
        ax.set_ylabel('Counts')
        ax.set_title(title)
        ax.set_xticks(numpy.arange(self.num_bins + 1))
        ax.set_xticklabels(xlabs)
        ax.grid()

        ax.legend([r[0] for r in rects], leg)

        fig.show()


    # the svg version 
    def get_svg_hist(self, y_scale=2.0, others=[], width=640, height=180, 
                     title='', y_label='', svg_id=''):

        # default plot settings
        margin = 40
        title_x = 40
        title_y = 30
        tick_height = 4
        tick_width = 4
        bar_margin = 3
        inner_bar_margin = 1
        
        # derived
        num_hists = 1 + len(others)
        x_orig = margin
        y_orig = height - margin
        x_max = width - margin
        y_max = margin
        binwidth = int((x_max - x_orig) / self.num_bins)
        bar_width = ((binwidth - 2 * bar_margin) / num_hists) - inner_bar_margin

        # obtain all hist data
        all_hist_data = [self.hist_data]
        for h in others:
            all_hist_data.append(h.hist_data)

        # first the svg style sheet
        result = '''
<svg class="svg_hist" id="%s" xmlns="http://www.w3.org/2000/svg" version="1.1">''' % (svg_id)
        result += '''
    <style type="text/css" >
    <![CDATA[
        rect.bar0 {
          fill: %s;
            }''' % (self.color)

        for hist_data_i in xrange(len(others)):
            result += '''
        rect.bar%i {
          fill: %s;
            }''' % (hist_data_i + 1, others[hist_data_i].color)

        result += '''
        rect.legend { 
          stroke-width: 0.5;
          stroke: #2e3436;
        }
        rect#pointer {
          fill: #cc0000;
        }
        line.axis {
          stroke: #888a85;
          stroke-width: 1px;
        }
        line.grid {
          stroke: #eeeeec;
          stroke-width: 1px;
        }
        text.axis {
          fill: #555753;
          font-size: 10px;
        }
        line#pred {
          stroke: #3465a4;
          stroke-width: 1px;
        }
        text#pred {
          fill: #3465a4;
        }
        text.title {
          fill: #2e3436;
          font-size: 13px;
        }
    ]]>
    </style>
    '''

        # add the title (change coordinates here if needed)
        result += '''
    <text class="title" x="%i" y="%i">%s</text>''' % (title_x, title_y, title)

        ########################################################################
        # Axes
        ########################################################################
        
        # draw xaxis
        result += '''    
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig, x_max, y_orig, y_orig)
        
        # draw xticks
        score = self.min_val
        while(score < self.max_val + 0.001): 
            xpos = margin + (x_max - x_orig) * (score / (self.max_val - self.min_val))
            result += '''    
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (xpos, xpos, y_orig, y_orig + tick_height)
            shift = 10 if score < 0.0 else 7
            result += '''
    <text class="axis" x="%i" y="%i">%.2f</text>''' % (xpos - shift, y_orig + 12, score)
            score += self.step_size
        
        # draw xlabel
        #result += '''
#    <text class="axis" x="%i" y="%i">%s</text>''' % ((width / 2) - 35, y_orig + 24, y_label)    

        # draw xaxis2
        result += '''    
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig, x_max, y_max, y_max)
        
        # draw yaxis
        result += '''
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig, x_orig, y_orig, y_max)
        
        # draw yaxis1 ticks and tick labels
        count = 0
        max_x = (100 * (1.0 / y_scale)) # TODO ??? even over nadenken...
        step_size = max_x / 8
        while(count < max_x + 1): 
            ypos = y_orig - (count * y_scale)
            result += '''    
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig - tick_width, x_orig, ypos, ypos)
            if not(count == 0 or count + step_size > max_x + 1): 
                result += '''
    <line class="grid" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_orig, x_max, ypos, ypos)
            shift = 25
            if(count < 100):
                shift = 18
            if(count < 10):
                shift = 12
            result += '''
    <text class="axis" x="%i" y="%i">%i</text>''' % (x_orig - shift, ypos + 3, count)
            count += step_size

        # draw ylabel1
        labx = x_orig - 32
        laby = (height / 2) + 25
        result += '''
    <text class="axis" x="%i" y="%i" transform="rotate(-90 %i, %i)">count</text>''' % (labx, laby, labx, laby)    

        # draw yaxis1 legend
        #ypos = y_max + 10
        #for index in xrange(len(pred_classes)):
        #    pc = pred_classes[index]
        #    leg = legend_strings[index]
        #    result += '''
        #    <rect x="%i" y="%i" width="%i" height="%i" class="%s legend"/>''' % (x_orig + 10, ypos, 10, 10, pc)
        #    result += '''
        #    <text class="axis" x="%i" y="%i">%s</text>''' % (x_orig + 25, ypos + 8, leg)
        #    ypos += 15

        # draw yaxis2
        result += '''
    <line class="axis" x1="%i" x2="%i" y1="%i" y2="%i"/>''' % (x_max, x_max, y_orig, y_max)

        # draw the histogram   
        for index in xrange(len(all_hist_data)):
            current_data = all_hist_data[index]

            # use the number of bins of this object...
            for bi in xrange(self.num_bins):
                h = int(round(current_data[bi] * y_scale))
                x = x_orig + bar_margin +  bi * binwidth + index * (bar_width + inner_bar_margin)
                y = y_orig - h
                if(h > 0):
                    result += '''        
    <rect class="bar%i" x="%i" y="%i" width="%i" height="%i"/>''' % (index, x, y, bar_width, h)
    
        # and finally the closing tag and it's 22:00, time to close the laptop.
        result += '''
</svg>
'''
        return result
