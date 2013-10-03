#!/usr/bin/env python

import os
import sys

# HACK TODO remove if sklearn is updated to 0.14 on compute servers...
import sklearn
if not(sklearn.__version__ == '0.14.1'):
    sys.path.insert(1, os.environ['SKL'])
    reload(sklearn)
assert(sklearn.__version__ == '0.14.1')

from sklearn.externals import joblib

from spice import classification
from spice import featmat
from biopy import file_io


def classify(fm_dir, cl_dir):
    '''
    PRE: required features are available in fe_dir!
    '''

    f_pre = os.path.basename(os.path.dirname(os.path.dirname(fm_dir)))

    # create dir to store the classification output and feature calc...
    out_dir = os.path.join(cl_dir, 'class_output')
    if not(os.path.exists(out_dir)):
        os.makedirs(out_dir)

    # read feature ids that were used to train the classifier
    cl_settings_f = os.path.join(cl_dir, 'settings.txt')
    settings_dict = file_io.read_settings_dict(cl_settings_f)
    feature_ids = settings_dict['feature_names']

    # obtain feature matrix STANDARDIZED DATA
    fm = featmat.FeatureMatrix.load_from_dir(fm_dir)
    feat_is = fm.feature_indices(feature_ids)
    object_is = range(len(fm.object_ids))
    data = fm.standardized_slice(feat_is, object_is)

    # load trained classifier
    cl_f = os.path.join(cl_dir, 'classifier.joblib.pkl')
    classifier = joblib.load(cl_f)

    # run classify method
    preds, probas = classification.classify(data, classifier)

    pred_f = os.path.join(out_dir, '%s_pred.txt' % (f_pre))
    proba_f = os.path.join(out_dir, '%s_proba.txt' % (f_pre))

    file_io.write_tuple_list(pred_f, zip(fm.object_ids, preds))
    file_io.write_tuple_list(proba_f, zip(fm.object_ids, probas))

#if __name__ == '__main__':
# TODO add test runs
