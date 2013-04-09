import os
import sys
import glob

import numpy
from matplotlib import pyplot
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial import distance
#import cairo
#import rsvg

# HACK TODO remove if sklearn is updated to 0.14
sys.path.insert(1, os.environ['SKL'])
assert(sklearn.__version__ == '0.14-git')
from sklearn.datasets.base import Bunch

from util import file_io
from spica.plotpy import histogram
from spica.plotpy import scatter
from spica.plotpy import heatmap
from spica.plotpy import color


class FeatureMatrix(object):

    # files used for data persistence
    objects_f = 'object_ids.txt'
    feature_matrix_f = 'feature_matrix.mat'
    features_f = 'feature_ids.txt'
    labels_f = 'labels.txt'

    def __init__(self):

        # init feature matrix to None, set with init
        self.object_ids = None
        self.feature_ids = None
        self.feature_matrix = None

        # define file location if we want to store data, set_root_dir
        self.root_dir = None

        # labelings
        self.labeling_dict = None

    #
    # Initialization functions
    #

    def set_object_ids(self, object_ids):

        # check and store object ids
        self._check_object_ids(object_ids)
        self.object_ids = object_ids

        # by default set one_class labeling
        label_dict = dict(zip(self.object_ids, [0] * len(self.object_ids)))
        self.add_labels('one_class', label_dict, ['all'])

    def load_object_ids(self, objects_f):
        with open(objects_f, 'r') as fin:
            object_ids = [i for i in file_io.read_ids(fin)]
        self.set_object_ids(object_ids)

    def load_labels(self, labeling_name, labeling_f):

        l = Labeling(labeling_name, self)
        l.load(labeling_f)
        self.add_labeling(l)

    def add_labels(self, labeling_name, label_dict, class_names):

        l = Labeling(labeling_name, self)
        l.set_labels(label_dict, class_names)
        self.add_labeling(l)

    def add_labeling(self, labeling):

        assert(type(labeling) == Labeling)

        if not(self.object_ids):
            raise ValueError('Objects are not set.')

        # store self as callback if not there yet
        if not(labeling.feature_matrix):
            labeling.set_feature_matrix(self)
        elif not(labeling.feature_matrix == self):
            raise ValueError('Labeling already linked to other feature matrix')

        if not(self.labeling_dict):
            self.labeling_dict = {}

        # add to labeling_dict
        self.labeling_dict[labeling.name] = labeling

    def add_features(self, feature_ids, feature_matrix):

        # check for errors
        self._check_features(feature_ids, feature_matrix)

        # store arguments
        if(self.feature_ids):
            self.feature_ids.extend(feature_ids)
            self.feature_matrix = numpy.hstack([self.feature_matrix,
                                                feature_matrix])
        else:
            self.feature_ids = feature_ids
            self.feature_matrix = feature_matrix

    #TODO remove_features

    def delete_feature_matrix(self):
        self.feature_matrix = None
        self.feature_ids = None

    def set_root_dir(self, root_dir):
        self.root_dir = root_dir
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        self.img_dir = os.path.join(self.root_dir, 'img')
        self.histogram_dir = os.path.join(self.img_dir, 'histogram')
        self.scatter_dir = os.path.join(self.img_dir, 'scatter')
        self.heatmap_dir = os.path.join(self.img_dir, 'heatmap')

    def slice(self, feat_is, object_is):
        data = self.feature_matrix[:, feat_is]
        return data[object_is, :]

    def standardized(self):
        return self._standardize(self.feature_matrix)

    def standardized_slice(self, feat_is, object_is):
        return self._standardize(self.slice(feat_is, object_is))

    def _standardize(self, mat):
        result = numpy.copy(mat)
        # column wise (features)
        mean = numpy.mean(result, axis=0)
        std = numpy.std(result, axis=0)
        # reset zeros to one, to avoid NaN
        std[std == 0.0] = 1.0
        result -= mean
        result /= std
        return result

    def feature_indices(self, feature_ids):
        # TODO
        '''
        Where do I use this for??? Looks like a mistake, obtaining all
        feature indices, and then sort them??? sounds like the same as
        range(len(self.feature_ids))
        '''
        return sorted([self.feature_ids.index(f) for f in feature_ids])

    def object_indices(self, labeling_name, class_ids):
        labeling = self.labeling_dict[labeling_name]
        indices = []
        for c in class_ids:
            indices.extend(labeling.object_indices_per_class[c])
        return sorted(indices)

    def class_indices(self, labeling_name, class_ids):
        labeling = self.labeling_dict[labeling_name]
        return sorted([labeling.class_names.index(c) for c in class_ids])

    def merge(self, other):

        # check if other has the some objects and labels (same order as well)
        if not(self.object_ids == other.object_ids):
            raise ValueError('Object of the two feature matrices ' +
                             'do not correspond.')
        if not(self.labels_list == other.labels_list):
            raise ValueError('Labels of the two feature matrices ' +
                             'do not correspond.')

        # add the feature ids and extend the feature matrix
        self.add_features(other.feature_ids, other.feature_matrix)

    def get_dataset(self, feat_ids=None, labeling_name=None, class_ids=None,
            standardized=True):

        if not(labeling_name):
            labeling_name = 'one_class'
        labeling = self.labeling_dict[labeling_name]

        if(feat_ids or class_ids):

            if not(feat_ids):
                feat_ids = self.feature_ids
            if not(class_ids):
                class_ids = labeling.class_names

            feat_is = self.feature_indices(feat_ids)
            object_is = self.object_indices(labeling_name, class_ids)
            class_is = self.class_indices(labeling_name, class_ids)
            if standardized:
                fm = self.standardized_slice(feat_is, object_is)
            else:
                fm = self.slice(feat_is, object_is)
            target = [labeling.labels[i] for i in object_is]
            target_names = [labeling.class_names[i] for i in class_is]
            sample_names = [self.object_ids[i] for i in object_is]
            feature_names = [self.feature_ids[i] for i in feat_is]

            # map target to use 0,1,2,... as labels
            target_map = dict(zip(class_is, range(len(class_is))))
            target = numpy.array([target_map[t] for t in target])
        else:
            if standardized:
                fm = self.standardized()
            else:
                fm = self.feature_matrix
            target = numpy.array(labeling.labels)
            target_names = labeling.class_names
            sample_names = self.object_ids
            feature_names = self.feature_ids

        return (fm, sample_names, feature_names, target, target_names)

    def get_sklearn_dataset(self, feat_ids=None, labeling_name=None,
            class_ids=None, standardized=True):

        (fm, sample_names, feature_names, target, target_names) =\
                self.get_dataset(feat_ids, labeling_name, class_ids,
                standardized)

        return Bunch(data=fm,
                     target=target,
                     target_names=target_names,
                     sample_names=sample_names,
                     feature_names=feature_names)
                     #DESCR='')# TODO

    #
    # Data storage functions
    #

    def load(self):
        self._check_root_dir_set()
        self._load_object_ids()
        self._load_feature_ids()
        self._load_feature_matrix()
        self._load_labels()

    def save(self):
        self._check_root_dir_set()
        if not(os.path.exists(self.root_dir)):
            os.makedirs(self.root_dir)
        self._store_object_ids()
        self._store_feature_ids()
        self._store_feature_matrix()
        self._store_labels()

    #
    # String representation
    #
    def __str__(self):

        s = '\nFeatureMatrix:\n\n'

        if(self.object_ids):
            s += 'object ids:\n%s\n\n' % (str(self.object_ids))
        if(self.feature_ids):
            s += 'feature ids:\n%s\n\n' % (str(self.feature_ids))
        # TODO add labelings
        if not(self.feature_matrix is None):
            s += 'feature matrix:\n%s\n\n' % (str(self.feature_matrix))

        return s

    # TODO __repr__

    #
    # Feature matrix based calculation / visualization functions
    #

    def ttest(self, labeling_name, label0, label1, object_is=None):

        ts = []

        if(self.feature_ids):

            try:
                labeling = self.labeling_dict[labeling_name]
            except KeyError:
                raise ValueError('Labeling does not exist: %s.' %
                        (labeling_name))
            try:
                obj_is_per_class = labeling.get_obj_is_per_class(object_is)
                lab0_indices = obj_is_per_class[label0]
                lab1_indices = obj_is_per_class[label1]
            except KeyError:
                raise ValueError('Non-existing label provided.')

            for f_i in xrange(len(self.feature_ids)):
                # TODO check for class variations, to determine if equal_val
                # should be set to False? In that case a Welch's t-test is
                # performed.
                #
                # Maybe for current situation equal sample sizes are assumed
                # as well. Check the scipy code and compare with the t-test
                # formulas on wikipedia.
                # DONE: looks like the unequal sample sizes, equal variance
                #       formula
                ts.append(stats.ttest_ind(
                        self.feature_matrix[lab0_indices, f_i],
                        self.feature_matrix[lab1_indices, f_i]))

        return ts

    def save_histogram(self, feat_id, labeling_name, labels=None, colors=None):

        try:
            labeling = self.labeling_dict[labeling_name]
        except KeyError:
            raise ValueError('Labeling does not exist: %s.' % (labeling_name))

        # use all labels by default
        if not(labels):
            labels = labeling.class_names

        if not(os.path.exists(self.histogram_dir)):
            os.makedirs(self.histogram_dir)
        #label_indices = [labeling.class_names.index(l) for l in labels]
        #class_length = [labeling.labels.count(i) for i in label_indices]
        #maxy = histogram.rounded_maxy(max(class_length) / 3)
        #maxy = histogram.rounded_maxy(max(class_length))

        # set some default colors, if None provided
        #if not(colors):
        #    colors = color.color_dict()

        try:
            feature_index = self.feature_ids.index(feat_id)
        except ValueError:
            raise ValueError('Feature %s does not exist.' % (feat_id))

        # standardize data
        fm = self.standardized()

        #feat_hists = []
        lab_str = labeling_name + '_' + '_'.join([str(l) for l in labels])
        out_f = os.path.join(self.histogram_dir,
                '%s_%s.png' % (feat_id, lab_str))

        hist_data = []
        for lab_i, lab in enumerate(labels):

            lab_indices = labeling.object_indices_per_class[lab]

            # fetch feature column with only the object rows with label lab
            h_data = fm[lab_indices, feature_index]
            hist_data.append(h_data)

            # create histogrom object
            #h = histogram.Histogram(h_data)
            #h.settings(color=colors[lab_i], maxy=maxy, legend=lab)
            #h.calc()
            #feat_hists.append(h)

        fig = pyplot.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(hist_data, bins=40, color=colors)
        ax.legend(labels)
        ax.grid()
        fig.savefig(out_f, bbox_inches='tight')

        pyplot.close(fig)

    def get_hist_object(self, feat_id, labeling_name, labels=None,
            colors=None):

        try:
            labeling = self.labeling_dict[labeling_name]
        except KeyError:
            raise ValueError('Labeling does not exist: %s.' % (labeling_name))

        # use all labels by default
        if not(labels):
            labels = labeling.class_names

        # TODO get length of largest class
        label_indices = [labeling.class_names.index(l) for l in labels]
        class_length = [labeling.labels.count(i) for i in label_indices]
        #maxy = histogram.rounded_maxy(max(class_length) / 3)
        maxy = histogram.rounded_maxy(max(class_length))

        # set some default colors, if None provided
        if not(colors):
            colors = color.color_dict()

        try:
            feature_index = self.feature_ids.index(feat_id)
        except ValueError:
            raise ValueError('Feature %s does not exist.' % (feat_id))

        # standardize data
        fm = self.standardized()

        feat_hists = []

        for lab_i, lab in enumerate(labels):

            lab_indices = labeling.object_indices_per_class[lab]

            # fetch feature column with only the object rows with label lab
            h_data = fm[lab_indices, feature_index]

            # create histogrom object
            h = histogram.Histogram(h_data)
            h.settings(color=colors[lab_i], maxy=maxy, legend=lab)
            h.calc()
            feat_hists.append(h)

        return feat_hists

    def get_hist_svg(self, feat_id, labeling_name=None, labels=None,
            colors=None):
        hists = self.get_hist_object(feat_id, labeling_name, labels, colors)
        return hists[0].get_norm_svg_hist(others=hists[1:],
                title=feat_id, svg_id=feat_id)

    def get_hist_svg_path(self, feat_id, labeling_name=None, labels=None):

        if not(os.path.exists(self.histogram_dir)):
            os.makedirs(self.histogram_dir)

        if not(labeling_name):
            labeling_name = self.labeling_dict[sorted(
                    self.labeling_dict.keys())[0]].name

        if not(labels):
            labels = self.labeling_dict[labeling_name].class_names

        lab_str = labeling_name + '_' + '_'.join([str(l) for l in labels])
        out_f = os.path.join(self.histogram_dir,
                '%s_%s.svg' % (feat_id, lab_str))

        if not(os.path.exists(out_f)):
            with open(out_f, 'w') as fout:
                fout.write(self.get_hist_svg(feat_id, labeling_name, labels))

        return out_f

    '''
    def get_hist_png_path(self, feat_id, labeling_name=None, labels=None,
            scale=1.0):

        # TODO implement scale

        svg_f = self.get_hist_svg_path(feat_id, labeling_name, labels)
        png_f = '%s.png' % (svg_f)

        if not(os.path.exists(png_f)):
            img = cairo.ImageSurface(cairo.FORMAT_ARGB32, 640, 210)
            ctx = cairo.Context(img)
            handler= rsvg.Handle(svg_f)
            handler.render_cairo(ctx)
            img.write_to_png(png_f)

        return png_f
    '''
    def save_scatter(self, feat_id0, feat_id1, labeling_name=None,
            classes=None):
        if not(os.path.exists(self.scatter_dir)):
            os.makedirs(self.scatter_dir)

        if not(labeling_name):
            labeling_name = self.labeling_dict[sorted(
                    self.labeling_dict.keys())[0]].name

        if not(classes):
            classes = self.labeling_dict[labeling_name].class_names

        try:
            labeling = self.labeling_dict[labeling_name]
        except KeyError:
            raise ValueError('Labeling does not exist: %s.' % (labeling_name))

        #if not(colors):
        colors = color.color_dict()

        try:
            feature_index0 = self.feature_ids.index(feat_id0)
            feature_index1 = self.feature_ids.index(feat_id1)
        except ValueError:
            raise ValueError('Feature %s or %s does not exist.' %
                    (feat_id0, feat_id1))

        # standardize data NOTE that fm is standardized before the objects
        # are sliced out!!!
        # not sure if this is the desired situation...
        fm = self.standardized()

        # slice two feature columns
        data = fm[:, [feature_index0, feature_index1]]

        # slice labels subdata
        subdata = []
        labels = []
        for index, c in enumerate(classes):
            #class_index = labeling.class_names.index(c)
            object_indices = labeling.object_indices_per_class[c]
            subdata.append(data[object_indices, :])
            labels.extend(len(object_indices) * [index])
        data = numpy.vstack(subdata)

        return scatter.Scatter(data, labels, feat_id0, feat_id1, classes)

    def get_scatter_object(self, feat_id0, feat_id1, labeling_name=None,
            classes=None, colors=None, standardized=True):

        if not(os.path.exists(self.scatter_dir)):
            os.makedirs(self.scatter_dir)

        if not(labeling_name):
            labeling_name = self.labeling_dict[sorted(
                    self.labeling_dict.keys())[0]].name

        if not(classes):
            classes = self.labeling_dict[labeling_name].class_names

        try:
            labeling = self.labeling_dict[labeling_name]
        except KeyError:
            raise ValueError('Labeling does not exist: %s.' % (labeling_name))

        if not(colors):
            colors = color.color_dict()

        try:
            feature_index0 = self.feature_ids.index(feat_id0)
            feature_index1 = self.feature_ids.index(feat_id1)
        except ValueError:
            raise ValueError('Feature %s or %s does not exist.' %
                    (feat_id0, feat_id1))

        # standardize data NOTE that fm is standardized before the objects
        # are sliced out!!!
        # not sure if this is the desired situation...
        print standardized
        if(standardized):
            print 'standardize'
            fm = self.standardized()
        else:
            fm = numpy.copy(self.feature_matrix)
        # slice two feature columns
        data = fm[:, [feature_index0, feature_index1]]
        print data

        # slice labels subdata
        subdata = []
        labels = []
        for index, c in enumerate(classes):
            #class_index = labeling.class_names.index(c)
            object_indices = labeling.object_indices_per_class[c]
            subdata.append(data[object_indices, :])
            labels.extend(len(object_indices) * [index])
        data = numpy.vstack(subdata)

        return scatter.Scatter(data, labels, feat_id0, feat_id1, classes)

    def get_scatter_svg(self, feat_id0, feat_id1, labeling_name=None,
            classes=None, colors=None, standardized=True):

        print standardized
        scat = self.get_scatter_object(feat_id0, feat_id1,
                labeling_name=labeling_name, classes=classes, colors=colors,
                standardized=standardized)
        return scat.get_norm_svg_scatter(title='%s_%s' % (feat_id0, feat_id1))

    def get_scatter_svg_path(self, feat_id0, feat_id1, labeling_name=None,
            classes=None, standardized=True):

        if not(os.path.exists(self.scatter_dir)):
            os.makedirs(self.scatter_dir)

        if not(labeling_name):
            labeling_name = self.labeling_dict[sorted(
                    self.labeling_dict.keys())[0]].name

        if not(classes):
            classes = self.labeling_dict[labeling_name].class_names

        lab_str = labeling_name + '_' + '_'.join([str(l) for l in classes])
        out_f = os.path.join(self.scatter_dir,
                '%s_%s_%s.svg' % (feat_id0, feat_id1, lab_str))

        if not(os.path.exists(out_f)):
            print standardized
            with open(out_f, 'w') as fout:
                fout.write(self.get_scatter_svg(feat_id0, feat_id1,
                        labeling_name=labeling_name, classes=classes,
                        standardized=standardized))

        return out_f

    # TODO create general function that creates paths to images...
    # TODO create general function that converts svg into png
    # TODO use width as parameter

    '''
    def get_scatter_png_path(self, feat_id0, feat_id1, labeling_name=None,
            classes=None, scale=1.0):

        # TODO implement scale

        svg_f = self.get_scatter_svg_path(feat_id0, feat_id1, labeling_name,
                classes)
        png_f = '%s.png' % (svg_f)

        if not(os.path.exists(png_f)):
            img = cairo.ImageSurface(cairo.FORMAT_ARGB32, 550, 550)
            ctx = cairo.Context(img)
            handler= rsvg.Handle(svg_f)
            handler.render_cairo(ctx)
            img.write_to_png(png_f)

        return png_f
    '''

    def get_clustdist_path(self, feature_ids=None, labeling_name=None,
                class_ids=None, vmin=-3.0, vmax=3.0):

        if not(os.path.exists(self.heatmap_dir)):
            os.makedirs(self.heatmap_dir)

        if not(labeling_name):
            labeling_name = 'one_class'
        #labeling = self.labeling_dict[labeling_name]

        (fm, sample_names, feature_names, target, target_names) =\
                self.get_dataset(feature_ids, labeling_name, class_ids)

        #fistr = '_'.join([str(self.feature_ids.index(f)) for f in
        #    feature_names])
        #listr = '_'.join([str(labeling.class_names.index(t))
        #        for t in target_names])
        #lab_str = 'feati_' + fistr + '_' + labeling_name + '_' + listr

        #png_f = os.path.join(self.heatmap_dir, 'fm_clustered_%s.png' %
        #        (lab_str))
        file_path = os.path.join(self.heatmap_dir, 'fm_clustered')

        # reorder feature matrix rows (objects)
        object_indices = hierarchy.leaves_list(self.clust_object(fm))
        fm = fm[object_indices, :]

        # reorder standardized feature matrix columns (feats)
        feat_indices = hierarchy.leaves_list(self.clust_feat(fm))
        fm = fm[:, feat_indices]

        # add labels of all available labelings (reordered using object_is)
        #lablists = [[l.labels[i] for i in object_indices]
        #                         for l in self.labeling_dict.values()
        #                         if not l.name == 'one_class']
        lablists = [[target[i] for i in object_indices]]
        class_names = [target_names]

        # reorder the feature and object ids
        fs = [feature_names[i] for i in feat_indices]
        gs = [sample_names for i in object_indices]

        heatmap.heatmap_labeled_fig(fm, fs, gs, lablists, class_names,
                                          file_path, vmin=vmin, vmax=vmax)

        return file_path

    def dist_feat(self, fm, metric='euclidian'):
        return self._dist(fm, 1, metric)

    #def dist_object(self, metric='euclidian'):
    #    return self._dist(fm, 0, metric)

    def _dist(self, fm, axis, metric):
        fm = fm.copy()
        if(axis == 1):
            fm = fm.transpose()
        # calculate and return dist matrix (condensed matrix as result!)
        return distance.pdist(fm)

    def clust_feat(self, fm, linkage='complete'):
        '''
        This function returns a hierarchical clustering of the features
        (columns) in the data matrix.
        Distance metric: euclidian
        Cluster distance metric: euclidian
        '''
        return self._clust(fm, 1, linkage)

    def clust_object(self, fm, linkage='complete'):
        return self._clust(fm, 0, linkage)

    def _clust(self, fm, axis, linkage):
        dist = self._dist(fm, axis, 'euclidian')
        return hierarchy.linkage(dist, method=linkage)

    def feature_correlation_matrix(self):
        return numpy.corrcoef(self.feature_matrix, rowvar=0)

    def feature_correlation_heatmap(self):
        if not(os.path.exists(self.heatmap_dir)):
            os.makedirs(self.heatmap_dir)
        f = os.path.join(self.heatmap_dir, 'feature_correlation.png')
        corr_matrix = self.feature_correlation_matrix()
        xlab = self.feature_ids
        ylab = self.feature_ids
        heatmap.heatmap_fig(corr_matrix, xlab, ylab, f, vmin=-1.0, vmax=1.0)
        return f
        
    #
    # Data storage help functions
    #

    def _store_object_ids(self):
        if(self.object_ids):
            f = os.path.join(self.root_dir, FeatureMatrix.objects_f)
            with open(f, 'w') as fout:
                file_io.write_ids(fout, self.object_ids)

    def _store_feature_ids(self):
        f = os.path.join(self.root_dir, FeatureMatrix.features_f)
        if not(self.feature_ids is None):
            with open(f, 'w') as fout:
                file_io.write_ids(fout, self.feature_ids)
        elif(os.path.exists(f)):
            os.remove(f)

    def _store_feature_matrix(self):
        f = os.path.join(self.root_dir, FeatureMatrix.feature_matrix_f)
        if not(self.feature_matrix is None):
            numpy.savetxt(f, self.feature_matrix)
        elif(os.path.exists(f)):
            os.remove(f)

    def _store_labels(self):
        if(self.labeling_dict):
            if not(os.path.exists(self.labels_dir)):
                os.makedirs(self.labels_dir)
            for key, value in self.labeling_dict.iteritems():
                value.save(os.path.join(self.labels_dir, '%s.txt' % (key)))

    def _load_object_ids(self):
        f = os.path.join(self.root_dir, FeatureMatrix.objects_f)
        if(os.path.exists(f)):
            self.load_object_ids(f)

    def _load_labels(self):
        if(os.path.exists(self.labels_dir)):
            for f in glob.glob(os.path.join(self.labels_dir, '*.txt')):
                name = os.path.splitext(os.path.basename(f))[0]
                l = Labeling(name, self)
                l.load(f)
                self.add_labeling(l)

    def _load_feature_ids(self):
        f = os.path.join(self.root_dir, FeatureMatrix.features_f)
        if(os.path.exists(f)):
            with open(f, 'r') as fin:
                self.feature_ids = [i for i in file_io.read_ids(fin)]

    def _load_feature_matrix(self):
        f = os.path.join(self.root_dir, FeatureMatrix.feature_matrix_f)
        if(os.path.exists(f)):
            self.feature_matrix = numpy.loadtxt(f)
            # in case of 1D matrix, reshape to single column 2D matrix
            fm_shape = self.feature_matrix.shape
            if(len(fm_shape) == 1):
                n = fm_shape[0]
                self.feature_matrix = self.feature_matrix.reshape((n, 1))

    #
    # Check for consistency functions
    #

    def _check_root_dir_set(self):
        if(self.root_dir is None):
            raise ValueError('The root directory has not been set.')

    def _check_object_ids(self, ids):
        if not(len(ids) == len(set(ids))):
            raise ValueError('There are multiple objects with the same id.')

    def _check_features(self, feature_ids, feature_matrix):

        # check for duplicate features in the newly added list of features
        if not(len(feature_ids) == len(set(feature_ids))):
            raise ValueError('The added features contain duplicate ids.')

        # when adding new features, check for overlap with existing features
        if not(self.feature_ids is None):
            inter = set(feature_ids) & set(self.feature_ids)
            if not(len(inter) == 0):
                raise ValueError('Feature ids %s already exist.' % (inter))

        # check if the number of rows corresponds to the number of objects
        if not(feature_matrix.shape[0] == len(self.object_ids)):
            raise ValueError('The number of rows in the feature matrix ' +
                             'does not correspond to the number of objects')

        # check if the number of features corresponds to the number fo feat ids
        if not(feature_matrix.shape[1] == len(feature_ids)):
            raise ValueError('The number of columns in the feature matrix ' +
                             'does not correspond to the number of ' +
                             'provided feature ids.')


class Labeling(object):

    def __init__(self, name, feature_matrix):

        self.name = name
        self.feature_matrix = feature_matrix

        self.labels = None
        self.class_names = None

        self.object_indices_per_class = None

    def set_class_names(self, names):
        '''
        Parameter names is a mapping {int: str}.
        '''

        if not(self.labels):
            raise ValueError('Cannot set label names, no labels set yet.')

        if not(len(names) > max(self.labels)):
            raise ValueError('Not enough label names provided.')

        if not(all([type(n) == str for n in names])):
            raise ValueError('Class names should be of type str.')

        self.class_names = names

    def set_labels(self, labels_dict, class_names):

        # check label values
        if not(all([type(i) == int for i in labels_dict.values()])):
            raise ValueError('Label values must be of type int.')

        # check if there is a label for all objects
        missing = set(self.feature_matrix.object_ids) - set(labels_dict.keys())
        if(missing):
            raise ValueError('Labels are missing: %s' % str(missing))

        # create list of labels in same order as object ids
        self.labels = [labels_dict[o] for o in self.feature_matrix.object_ids]

        # set the label names
        self.set_class_names(class_names)

        # create dict with object indices per label
        self.object_indices_per_class = {}
        for index, o in enumerate(self.feature_matrix.object_ids):
            l = self.labels[index]
            self.object_indices_per_class.setdefault(self.class_names[l],
                    []).append(index)

    def get_label_dict(self):
        return dict(zip(self.feature_matrix.object_ids, self.labels))

    def get_obj_is_per_class(self, object_is=None):

        if(object_is is None):
            return self.object_indices_per_class
        else:
            obj_is = {}
            for object_i in object_is:
                cl = self.class_names[self.labels[object_i]]
                obj_is.setdefault(cl, []).append(object_i)
            return obj_is

    def load(self, labels_f):
        (label_dict, class_names) = file_io.read_labeling(labels_f)
        self.set_labels(label_dict, class_names)

    def save(self, labels_f):
        file_io.write_labeling(labels_f, self.feature_matrix.object_ids,
                self.labels, self.class_names)
