"""
.. module:: featmat

.. moduleauthor:: Bastiaan van den Berg <b.a.vandenberg@gmail.com>

"""

import os
#import sys
import glob
import json

import numpy
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial import distance
from matplotlib import pyplot

# HACK TODO remove if sklearn is updated to 0.14 on compute servers...
#import sklearn
#if not(sklearn.__version__ == '0.14.1'):
#    sys.path.insert(1, os.environ['SKL'])
#    reload(sklearn)
#assert(sklearn.__version__ == '0.14.1')

from sklearn.datasets.base import Bunch

from biopy import file_io
from spice.plotpy import heatmap


class FeatureMatrix(object):
    """This class is used to manage a feature matrix.

    The FeatureMatrix object manages an *m* x *n* matrix in which *m* is the
    number of objects (rows) and *n* the number of features (columns).

    The feature matrix is initiated as an empty matrix. First the objects
    (`object_ids`) need to be set. These cannot be altered afterwards or
    a ValueError will be raised. This is to make life a bit easier for now.
    Thus far, for our functionality, there is no need to modify the objects
    after creation of the feature matrix.

    When the objects are set, the `add_features` function can be used to add
    features and thereby fill the feature matrix with values. A list with
    feature ids and a feature matrix need are required for this, in which the
    provided matrix is a numpy matrix with feature values. The rows of this
    matrix should have the same order as the object ids in the FeatureMatrix
    object (this is a bit tricky, how could this be improved?). Optionally
    a list of feature names could also be provided.

    The `feature_matrix` and `feature_ids` variable cannot be set directly,
    features can only be added throug the `add_features` function. Both
    variables can be deleted using the default `del` function. However, to
    guarantee a consistent state, as soon as one of the two variables is
    deleted, the other will be deleted as well (as well as the feature names).

    Zero or more `Labeling` objects can be attatched to the feature matrix.

    """

    # labeling name and class name of the default one-class labeling
    ONE_CLASS_LABELING = 'one_class'
    ONE_CLASS_LABEL = 'all'

    # file names and directory structure used when saving a feature matrix
    OBJECT_IDS_F = 'object_ids.txt'
    FEATURE_MATRIX_F = 'feature_matrix.mat'
    FEATURE_IDS_F = 'feature_ids.txt'
    FEATURE_NAMES_F = 'feature_names.txt'
    LABELING_D = 'labels'
    IMG_D = 'img'
    HISTOGRAM_D = os.path.join(IMG_D, 'histogram')
    SCATTER_D = os.path.join(IMG_D, 'scatter')
    HEATMAP_D = os.path.join(IMG_D, 'heatmap')

    # default name prefix for added features without feature id/name
    CUSTOM_FEAT_PRE = 'cus'
    CUSTOM_FEAT_NAME = 'Custom feature vector'

    def __init__(self):

        # The feature matrix, object ids (rows), and feature ids (columns)
        self._feature_matrix = None
        self._object_ids = None
        self._feature_ids = []

        # optional feature annotation
        self._feature_names = {}

        # labelings
        self._labeling_dict = {}

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def feature_ids(self):
        return self._feature_ids

    @feature_ids.deleter
    def feature_ids(self):
        self._delete_all_features()

    @property
    def feature_matrix(self):
        return self._feature_matrix

    @feature_matrix.deleter
    def feature_matrix(self):
        self._delete_all_features()

    def _delete_all_features(self):
        self._feature_matrix = None
        self._feature_ids = []
        self._feature_names = {}

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.deleter
    def feature_names(self):
        self.feature_names = dict(zip(self.feature_ids, self.feature_ids))

    @property
    def labeling_dict(self):
        return self._labeling_dict

    @object_ids.setter
    def object_ids(self, object_ids):
        '''This function sets the object ids (feature matrix rows).

        Args:
            object_ids ([str]): The list with unique object ids.

        Raises:
            ValueError: If object_ids contains duplicates.
            ValueError: If object_ids is empty.
            ValueError: If the object ids are already set.

        This function sets the list of object ids which are the rows of the
        feature matrix. It is not allowed to have duplicate ids in object_ids,
        a ValueError will be raised in this case.

        The object ids can only be set once, a ValueError will be raised if
        this function is called while a list of objects is allready available.

        As soon as the objects are set, a (one class) labeling is created in
        which all objects obtain the same label.
        '''

        # check if the objects are allready set
        if not(self.object_ids is None):
            raise ValueError('Object ids are allready set.')

        # check if there are ids in the list
        if(len(object_ids) == 0):
            raise ValueError('The object ids list is empty.')

        # check and store object ids
        if not(len(object_ids) == len(set(object_ids))):
            raise ValueError('The list of object ids contains duplicates.')

        self._object_ids = object_ids

        # by default set one_class labeling
        label_dict = dict(zip(self._object_ids, [0] * len(self._object_ids)))
        self.add_labeling(self.ONE_CLASS_LABELING, label_dict,
                          [self.ONE_CLASS_LABEL])

    def load_object_ids(self, object_ids_f):
        '''
        This function reads ids from files and sets them as the object ids.

        Args:
            object_ids_f (str or file): The ids file.

        Raises:
            FileIOError: If the file does not exist.
        '''
        with open(object_ids_f, 'r') as fin:
            ids = [i for i in file_io.read_ids(fin)]
        self.object_ids = ids

    def add_labeling(self, labeling_name, label_dict, class_names):
        '''
        This function adds a labeling to the feature matrix.

        Args:
            | **labeling_name** *(str)*: The name of the labeling.
            | **label_dict** *(dict)*: An object_id to label mapping.
            | **class_names** *([str])*: A list with class names
        Raises:
            | **ValueError**: If a labeling with the same name already exists.
            | **ValueError**: If the labeling object ids do not correspond to
                              the object ids of this feature matrix
        '''
        if(labeling_name in self.labeling_dict.keys()):
            raise ValueError('A labeling with the same name already exists.')

        if not(set(self.object_ids).issubset(set(label_dict.keys()))):
            raise ValueError('Not every object has a label in this labeling')

        # get labels in the same order as our object ids
        labels = [label_dict[oid] for oid in self.object_ids]

        # create labeling object
        l = Labeling(labeling_name, self.object_ids, labels, class_names)

        # add labeling
        self.labeling_dict[labeling_name] = l

    def add_labeling_from_file(self, labeling_name, labeling_f):
        '''
        This function loads labelinf from a file and adds it as to the feature
        matrix.

        Args:
            labeling_name (str): The name of the labeling
            labeling_f (str or file): The labeling file
        '''
        l = Labeling.load_from_file(labeling_name, labeling_f)
        self.add_labeling(l.name, l.label_dict, l.class_names)

    def add_features(self, feature_ids, feature_matrix, feature_names=None):
        '''
        This function extends the feature matrix, adding the provided features.

        It is the users responsebility that the rows of the feature matrix are
        in the same order as the object ids. TODO how to improve this? Use
        merge instead?

        Args:
            feature_ids ([str]): List with feature ids.
            feature_matrix (numpy.array): The feature values.

        Kwargs:
            feature_names ([str]): Optional list of feature names.

        Raises:
            ValueError: If feature_ids contains duplicates.
            ValueError: If any of the feature ids already exists
            ValueError: If the feature matrix row count does not correspond to
                        the number of objects
            ValueError: If the feature matrix column count does not correspond
                        to the number of feature ids.
        '''

        # check for errors
        #self._check_features(feature_ids, feature_matrix)
        #def _check_features(self, feature_ids, feature_matrix):

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

        # check if the number of features corresponds to the number of feat ids
        if not(feature_matrix.shape[1] == len(feature_ids)):
            raise ValueError('The number of columns in the feature matrix ' +
                             'does not correspond to the number of ' +
                             'provided feature ids.')

        # append feature ids
        self.feature_ids.extend(feature_ids)
        # create feature matrix or append to feature matrix
        if(self.feature_matrix is None):
            self._feature_matrix = feature_matrix
        else:
            self._feature_matrix = numpy.hstack([self._feature_matrix,
                                                feature_matrix])

        # create feature id to name mapping
        if(feature_names is None):
            feat_name_dict = dict(zip(feature_ids, feature_ids))
        else:
            feat_name_dict = dict(zip(feature_ids, feature_names))

        # add feature names
        self.feature_names.update(feat_name_dict)

    def remove_features(self, feature_ids):
        '''
        This function removes the feature with id feat_id from the feature
        matrix.

        Args:
            feature_ids ([str]): List with feature ids.
        Raises:
            ValueError: If one of the feature_ids does not exist in this
                        feature matrix
        '''
        try:

            # get the column indices of the provided feature ids
            fis = self.feature_indices(feature_ids)

            # if all feature ids are given, use the feature matrix deleter
            if(len(fis) == len(self.feature_ids)):
                del self.feature_matrix
            else:
                # otherwise delete columns from feature matrix
                self._feature_matrix = numpy.delete(self.feature_matrix,
                                                    fis, 1)

                # and delete feature ids and names
                for fid in feature_ids:
                    self.feature_ids.remove(fid)
                    del self.feature_names[fid]

        except ValueError:
            raise ValueError('Feature id not in the feature matrix.')

    def merge(self, other):

        # check if other has the same objects and labels (same order as well)
        if not(self.object_ids == other.object_ids):
            raise ValueError('Object of the two feature matrices ' +
                             'do not correspond.')

        # TODO merge labelings???

        # add the feature ids and extend the feature matrix
        self.add_features(other.feature_ids, other.feature_matrix)

    def add_custom_features(self, feature_matrix):
        '''
        Rename this... is the same as add_features, but without supplying
        feature_ids (names). So maybe combine the two and turn feature_ids
        into a kwargs which defaults to None.
        '''

        num_obj, num_feat = feature_matrix.shape

        if not(num_obj == len(self.object_ids)):
            raise ValueError('Number of feature matrix rows does not '
                             'correspond to number of objects.')

        cust_feats = self.get_custom_features().values()
        cust_feats = [c[0].split('_')[0] for c in cust_feats]

        if(len(cust_feats) == 0):
            new_cust_feat_i = 0
        else:
            last_cust_feat = sorted(cust_feats)[-1]
            new_cust_feat_i = int(
                last_cust_feat[(len(self.CUSTOM_FEAT_PRE)):]) + 1

        featvec_id = '%s%i' % (self.CUSTOM_FEAT_PRE, new_cust_feat_i)
        feat_ids = ['%s_%i' % (featvec_id, i) for i in xrange(num_feat)]
        feat_names = ['%s %i - %i' % (self.CUSTOM_FEAT_NAME, new_cust_feat_i,
                                      i) for i in xrange(num_feat)]
        self.add_features(feat_ids, feature_matrix, feature_names=feat_names)

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
        '''
        This function returns the feature matrix column indices where the
        features with the provided ids can be found.

        Args:
            feature_ids ([str]): List with feature ids.
        Returns:
            list with column indices.
        Raises:
            ValueError: if one of the feature_ids is not in the list.
        '''
        return [self.feature_ids.index(fid) for fid in feature_ids]

    def object_indices(self, object_ids):
        '''
        This function returns the feature matrix row aindices where the objects
        with the provided ids can be found.

        Args:
            object_ids ([str]): List with object ids.
        Returns:
            list with row indices.
        Raises:
            ValueError: if one of the object_ids is not in the list.
        '''
        return [self.object_ids.index(oid) for oid in object_ids]

    def filtered_object_indices(self, labeling_name, class_ids):
        labeling = self.labeling_dict[labeling_name]
        indices = []
        for c in class_ids:
            indices.extend(labeling.object_indices_per_class[c])
        return sorted(indices)

    def class_indices(self, labeling_name, class_ids):
        labeling = self.labeling_dict[labeling_name]
        return sorted([labeling.class_names.index(c) for c in class_ids])

    def get_custom_features(self):
        '''
        This function returns the available custom feature vector ids.

        Returns a dictionary with the custom feature vector ids as keys and the
        number of features in this vector as value. Custom feature vectors are
        named cus0 (the next would be cus1) and the features are named cus0_0,
        cus0_1, ..., cus_0_5. If this would be the only custom feature vector,
        the function returns {'cus0': ['cus0_0', 'cus0_1', ...]}
        '''
        feat_dict = {}
        if(self.feature_ids):
            for fid in self.feature_ids:
                if(fid[:len(self.CUSTOM_FEAT_PRE)] == self.CUSTOM_FEAT_PRE):
                    pre = fid.split('_')[0]
                    feat_dict.setdefault(pre, []).append(fid)
        return feat_dict

    def get_dataset(self, feat_ids=None, labeling_name=None, class_ids=None,
                    standardized=True):

        if (labeling_name is None):
            labeling_name = 'one_class'
        labeling = self.labeling_dict[labeling_name]

        if(feat_ids or class_ids):

            if not(feat_ids):
                feat_ids = self.feature_ids
            if not(class_ids):
                class_ids = labeling.class_names

            feat_is = sorted(self.feature_indices(feat_ids))
            object_is = self.filtered_object_indices(labeling_name, class_ids)
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

            # targets are floats because liblinear classification wants this...
            target = numpy.array([float(target_map[t]) for t in target])
        else:
            if standardized:
                fm = self.standardized()
            else:
                fm = self.feature_matrix
            target = numpy.array([float(l) for l in labeling.labels])
            target_names = labeling.class_names
            sample_names = self.object_ids
            feature_names = self.feature_ids

        return (fm, sample_names, feature_names, target, target_names)

    def get_sklearn_dataset(self, feat_ids=None, labeling_name=None,
                            class_ids=None, standardized=True):

        (fm, sample_names, feature_names, target, target_names) =\
            self.get_dataset(feat_ids, labeling_name, class_ids, standardized)

        return Bunch(data=fm,
                     target=target,
                     target_names=target_names,
                     sample_names=sample_names,
                     feature_names=feature_names)
                     #DESCR='')# TODO

    @classmethod
    def load_from_dir(cls, d):
        '''
        This class method returns a FeatureMatrix object that has been
        constructed using data loaded from a feature matrix directory.

        Args:
            | **d** *(str)*: The path to the feature matrix directory.
        Raises:

        '''
        # initilaze empty feature matrix object
        fm = cls()

        # first load object ids, if available
        f = os.path.join(d, cls.OBJECT_IDS_F)
        if(os.path.exists(f)):
            fm.load_object_ids(f)

            # read and add labelings
            lab_d = os.path.join(d, cls.LABELING_D)
            if(os.path.exists(lab_d)):
                for f in glob.glob(os.path.join(lab_d, '*.txt')):
                    lname = os.path.splitext(os.path.basename(f))[0]
                    if not(lname == cls.ONE_CLASS_LABELING):
                        (label_dict, class_names) = file_io.read_labeling(f)
                        fm.add_labeling(lname, label_dict, class_names)

            fids = None
            fnames = None
            featmat = None

            # read feature ids
            f = os.path.join(d, cls.FEATURE_IDS_F)
            if(os.path.exists(f)):
                with open(f, 'r') as fin:
                    fids = [i for i in file_io.read_ids(fin)]

            # read feature names
            f = os.path.join(d, cls.FEATURE_NAMES_F)
            if(os.path.exists(f)):
                with open(f, 'r') as fin:
                    fnames = [n for n in file_io.read_names(fin)]

            # read feature matrix
            f = os.path.join(d, cls.FEATURE_MATRIX_F)
            if(os.path.exists(f)):
                featmat = numpy.loadtxt(f)
                # in case of 1D matrix, reshape to single column 2D matrix
                fm_shape = featmat.shape
                if(len(fm_shape) == 1):
                    n = fm_shape[0]
                    featmat = featmat.reshape((n, 1))

            if not(featmat is None):
                fm.add_features(fids, featmat, fnames)

        return fm

    def save_to_dir(self, d):
        '''
        This function stores the current feature matrix object to directory.

        Args:
            | **d** *(str)*: The path to the directory where the feature matrix
                             data will be stored.
        Raises:

        '''
        if not(os.path.exists(d)):
            os.makedirs(d)
        self._save_object_ids(os.path.join(d, self.OBJECT_IDS_F))
        self._save_feature_ids(os.path.join(d, self.FEATURE_IDS_F))
        self._save_feature_names(os.path.join(d, self.FEATURE_NAMES_F))
        self._save_feature_matrix(os.path.join(d, self.FEATURE_MATRIX_F))
        self._save_labelings(os.path.join(d, self.LABELING_D))

    def _save_object_ids(self, f):
        if(self.object_ids):
            with open(f, 'w') as fout:
                file_io.write_ids(fout, self.object_ids)

    def _save_feature_ids(self, f):
        if not(self.feature_ids is None):
            with open(f, 'w') as fout:
                file_io.write_ids(fout, self.feature_ids)
        elif(os.path.exists(f)):
            os.remove(f)

    def _save_feature_names(self, f):
        if not(self.feature_names is None):
            feat_names = [self.feature_names[fid] for fid in self.feature_ids]
            with open(f, 'w') as fout:
                file_io.write_names(fout, feat_names)
        elif(os.path.exists(f)):
            os.remove(f)

    def _save_feature_matrix(self, f):
        if not(self.feature_matrix is None):
            numpy.savetxt(f, self.feature_matrix, fmt='%.4e')
        elif(os.path.exists(f)):
            os.remove(f)

    def _save_labelings(self, d):
        if(self.labeling_dict):
            if not(os.path.exists(d)):
                os.makedirs(d)
            for lname, l in self.labeling_dict.iteritems():
                f = os.path.join(d, '%s.txt' % (lname))
                file_io.write_labeling(f, self.object_ids, l.labels,
                                       l.class_names)

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
                          self.feature_matrix[lab1_indices, f_i],
                          self.feature_matrix[lab0_indices, f_i]))

        return ts

    def histogram_data(self, feat_id, labeling_name, class_ids=None,
                       num_bins=40, standardized=False, title=None):

        # test num_bins > 0

        if(title is None):
            title = ''

        # get labeling data
        try:
            labeling = self.labeling_dict[labeling_name]
        except KeyError:
            raise ValueError('Labeling does not exist: %s.' % (labeling_name))

        # by default use all classes
        if not(class_ids):
            class_ids = labeling.class_names

        # get the feature matrix column index for the given feature id
        try:
            feature_index = self.feature_ids.index(feat_id)
        except ValueError:
            raise ValueError('Feature %s does not exist.' % (feat_id))

        # get the name of the feature
        feat_name = self.feature_names[feat_id]

        # get the feature matrix, standardize data if requested
        if(standardized):
            fm = self.standardized()
        else:
            fm = self.feature_matrix

        # generate histogram data
        hist_data = {}
        # TODO check what is the proper python way to this
        min_val = 10000.0
        max_val = -10000.0

        for lab in class_ids:

            lab_indices = labeling.object_indices_per_class[lab]

            # fetch feature column with only the object rows with label lab
            h_data = fm[lab_indices, feature_index]

            min_val = min(min_val, min(h_data))
            max_val = max(max_val, max(h_data))
            hist_data[lab] = h_data

        # round step size
        # quick and dirty, there's probably some elegant way to do this
        step = (max_val - min_val) / num_bins
        orde = 0
        tmp_step = step
        while(tmp_step < 1.0):
            orde += 1
            tmp_step *= 10
        orde = 10**orde
        step = round(step * orde) / orde

        # quick and dirty again, rounded range with nice bin boundaries
        start = 0.0

        if(min_val < 0.0):
            while(start > min_val):
                start -= step
    
        elif(min_val > 0.0):
            while(start < min_val):
                start += step
            start -= step

        end = 0.0

        if(max_val < 0.0):
            while(end > max_val):
                end -= step
            end += step

        elif(max_val > 0.0):
            while(end < max_val):
                end += step

        # generate the bin edges
        bin_edges = list(numpy.arange(start, end, step))

        max_count = 0
        hists = {}
        for lab in hist_data.keys():

            h, e = numpy.histogram(hist_data[lab], bin_edges)
            hists[lab] = list(h)

            max_count = max(max_count, max(h))

        # and again, quick and dirty, the y grid
        y_grid = []

        if(max_count < 100):
            t = max_count / 10
            y_grid = range(0, t * 10 + 1, t)
        elif(max_count < 1000):
            t = (max_count / 100) + 1
            y_grid = range(0, t * 100 + 1, t * 10)
        elif(max_count < 10000):
            t = (max_count / 1000) + 1
            y_grid = range(0, t * 1000 + 1, t * 100)
        elif(max_count < 100000):
            t = (max_count / 10000) + 1
            y_grid = range(0, t * 10000 + 1, t * 1000)
        else:
            y_grid = range(0, max_count, max_count / 10)
            
            

        result = {}
        result['feature-id'] = feat_id
        result['title'] = title
        result['x-label'] = feat_name
        result['legend'] = class_ids
        for lab in class_ids:
            result[lab] = hists[lab]
        result['min-value'] = min_val
        result['max-value'] = max_val
        result['max-count'] = max_count
        result['bin-edges'] = bin_edges
        result['y-grid'] = y_grid

        return result

    def histogram_json(self, feat_id, labeling_name, class_ids=None,
                       num_bins=40, standardized=False, title=None):

        if(title is None):
            title = ''

        hist_data = self.histogram_data(feat_id, labeling_name, class_ids,
                                        standardized=standardized, title=title)
        return json.dumps(hist_data)

    def save_histogram(self, feat_id, labeling_name, class_ids=None,
                       colors=None, img_format='png', root_dir='.',
                       title=None, standardized=False):

        try:
            labeling = self.labeling_dict[labeling_name]
        except KeyError:
            raise ValueError('Labeling does not exist: %s.' % (labeling_name))

        if(colors is None):
            colors = ['#3465a4', '#73d216', '#f57900', '#5c3566', '#c17d11',
                      '#729fcf', '#4e9a06', '#fcaf3e', '#ad7fa8', '#8f5902']

        # use all labels by default
        if not(class_ids):
            class_ids = labeling.class_names

        try:
            feature_index = self.feature_ids.index(feat_id)
        except ValueError:
            raise ValueError('Feature %s does not exist.' % (feat_id))

        feat_name = self.feature_names[feat_id]

        # standardize data
        if(standardized):
            fm = self.standardized()
        else:
            fm = self.feature_matrix

        #feat_hists = []
        lab_str = labeling_name + '_' + '_'.join([str(l) for l in class_ids])

        d = os.path.join(root_dir, self.HISTOGRAM_D)
        if not(os.path.exists(d)):
            os.makedirs(d)
        out_f = os.path.join(d, '%s_%s.%s' % (feat_id, lab_str, img_format))

        hist_data = []
        for lab_i, lab in enumerate(class_ids):

            lab_indices = labeling.object_indices_per_class[lab]

            # fetch feature column with only the object rows with label lab
            h_data = fm[lab_indices, feature_index]
            hist_data.append(h_data)

        fig = pyplot.figure(figsize=(8.8, 2.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(hist_data, bins=40, color=colors[:len(class_ids)])
        ax.set_xlabel(feat_name)
        ax.legend(class_ids)
        ax.grid()
        if(title):
            ax.set_title(title)
        fig.savefig(out_f, bbox_inches='tight')

        pyplot.close(fig)

        return out_f

    def save_scatter(self, feat_id0, feat_id1, labeling_name=None,
                     class_ids=None, colors=None, img_format='png',
                     root_dir='.', feat0_pre=None, feat1_pre=None,
                     standardized=False):

        try:
            labeling = self.labeling_dict[labeling_name]
        except KeyError:
            raise ValueError('Labeling does not exist: %s.' % (labeling_name))

        if(colors is None):
            colors = ['#3465a4', '#edd400', '#73d216', '#f57900', '#5c3566',
                      '#c17d11', '#729fcf', '#4e9a06', '#fcaf3e', '#ad7fa8',
                      '#8f5902']

        if not(labeling_name):
            labeling_name = self.labeling_dict[sorted(
                self.labeling_dict.keys())[0]].name

        if not(class_ids):
            class_ids = self.labeling_dict[labeling_name].class_names

        try:
            feature_index0 = self.feature_ids.index(feat_id0)
            feature_index1 = self.feature_ids.index(feat_id1)
        except ValueError:
            raise ValueError('Feature %s or %s does not exist.' %
                             (feat_id0, feat_id1))

        feat_name0 = self.feature_names[feat_id0]
        feat_name1 = self.feature_names[feat_id1]

        if(feat0_pre):
            feat_name0 = ' - '.join([feat0_pre, feat_name0])
        if(feat1_pre):
            feat_name1 = ' - '.join([feat1_pre, feat_name1])

        d = os.path.join(root_dir, self.SCATTER_D)
        if not(os.path.exists(d)):
            os.makedirs(d)
        out_f = os.path.join(d, 'scatter.%s' % (img_format))

        if(standardized):
            # standardize data NOTE that fm is standardized before the objects
            # are sliced out!!!
            # not sure if this is the desired situation...
            fm = self.standardized()
        else:
            fm = self.feature_matrix

        fig = pyplot.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)

        # for each class id, add object ids that have that class label
        for index, class_id in enumerate(class_ids):
            object_is = labeling.object_indices_per_class[class_id]
            x = fm[object_is, feature_index0]
            y = fm[object_is, feature_index1]
            c = colors[index]
            ax.scatter(x, y, s=30, c=c, marker='o', label=class_id)

        ax.set_xlabel(feat_name0)
        ax.set_ylabel(feat_name1)
        ax.legend(loc='upper right')
        ax.grid()
        fig.savefig(out_f, bbox_inches='tight')

        pyplot.close(fig)

        return out_f

    def get_clustdist_path(self, feature_ids=None, labeling_name=None,
                           class_ids=None, vmin=-3.0, vmax=3.0, root_dir='.'):

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
        d = os.path.join(root_dir, self.HEATMAP_D)
        if not(os.path.exists(d)):
            os.makedirs(d)
        img_format = 'png'
        file_path = os.path.join(d, 'fm_clustered.%s' % (img_format))

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
        if not(os.path.exists(self.HEATMAP_D)):
            os.makedirs(self.HEATMAP_D)
        f = os.path.join(self.HEATMAP_D, 'feature_correlation.png')
        corr_matrix = self.feature_correlation_matrix()
        xlab = self.feature_ids
        ylab = self.feature_ids
        heatmap.heatmap_fig(corr_matrix, xlab, ylab, f, vmin=-1.0, vmax=1.0)
        return f


class Labeling(object):

    #def __init__(self, name, feature_matrix):
    def __init__(self, name, object_ids, labels, class_names):
        '''
        Is it really necesary to retain the order of the object ids? Why not
        initiate with a dict?
        '''

        label_set = set(labels)

        if not(len(object_ids) == len(labels)):
            raise ValueError('Number of object ids and labels is different.')
        # ??? should I add this ???
        if not(label_set == set(range(len(label_set)))):
            raise ValueError('Labels should be 0, 1, ...')
        if not(len(label_set) == len(class_names)):
            raise ValueError('Number of class names does not correspond to ' +
                             'the number of different labels.')

        self._name = name
        self._object_ids = object_ids
        self._labels = labels
        self._label_dict = dict(zip(object_ids, labels))
        self._class_names = class_names

        self._object_indices_per_class = {}
        for index, o in enumerate(self._object_ids):
            c = self._class_names[self.labels[index]]
            self._object_indices_per_class.setdefault(c, []).append(index)

    @property
    def name(self):
        return self._name

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def labels(self):
        return self._labels

    @property
    def label_dict(self):
        return self._label_dict

    @property
    def class_names(self):
        return self._class_names

    @property
    def object_indices_per_class(self):
        return self._object_indices_per_class

    # replaced by label_dict property
    #def get_label_dict(self):
    #    return dict(zip(self.feature_matrix.object_ids, self.labels))

    def get_obj_is_per_class(self, object_is=None):
        '''
        This is for object subsets, who uses this???
        '''
        if(object_is is None):
            return self.object_indices_per_class
        else:
            obj_is = {}
            for object_i in object_is:
                cl = self.class_names[self.labels[object_i]]
                obj_is.setdefault(cl, []).append(object_i)
            return obj_is

    @classmethod
    def load_from_file(cls, labeling_name, f):
        (label_dict, class_names) = file_io.read_labeling(f)
        object_ids = sorted(label_dict.keys())
        labels = [label_dict[oid] for oid in object_ids]
        return cls(labeling_name, object_ids, labels, class_names)
