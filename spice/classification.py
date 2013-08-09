#!/usr/bin/env python

import os
import sys
import operator
import argparse
import time
import warnings
import traceback

import numpy
#from matplotlib import pyplot

# HACK TODO remove if sklearn is updated to 0.14
sys.path.insert(1, os.environ['SKL'])
import sklearn
assert(sklearn.__version__ == '0.14-git')
from sklearn import svm
from sklearn import neighbors
from sklearn import lda
from sklearn import qda
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.externals import joblib

from spice import featmat
#from biopy import timeout
from biopy import roc
from biopy import file_io

# classification performance measures
all_score_names = ['roc_auc', 'f1', 'precision', 'average_precision',
                   'recall', 'accuracy']
all_score_funcs = dict(zip(all_score_names, [
    metrics.auc_score,
    metrics.f1_score,
    metrics.precision_score,
    metrics.average_precision_score,
    metrics.recall_score,
    metrics.accuracy_score]))
all_score_input = {'pred': ['f1', 'precision', 'recall', 'accuracy'],
                   'proba': ['roc_auc', 'average_precision']}

# minimal and maximal score per score measure
# TODO extend, only used in ffs now
metric_rand_max_score = {
    'roc_auc': (0.5, 1.0),
    'f1': (0.0, 1.0)
}

# default classifier parameters
svm_default_param = {'class_weight': 'auto'}
libsvm_default_param = {'class_weight': 'auto', 'probability': True,
                        'cache_size': 500.0}
rn_default_param = {'outlier_label': 0}
kn_default_param = {}

# which parameter is used by what classifier
classifier_params = ['gamma', 'C', 'n_neighbors', 'radius']
classifiers_per_param = {
    'C': ['linearsvc', 'svc_linear', 'svc_rbf'],
    'gamma': ['svc_rbf'],
    'n_neighbors': ['kn_uniform', 'kn_distance'],
    'radius': ['rn_uniform', 'rn_distance']
}

# default paramet ranges
default_param_range = {
    'n_neighbors': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100],
    'radius': range(1, 11),
    'C': 10.0 ** numpy.arange(-5, 6),
    'gamma': 10.0 ** numpy.arange(-6, 3)
}

# timed parameters
timed_param = ['C', 'radius', 'n_neighbors']

#
# methods for scaled data
#


def cv_scores_no_scaling(data, target, classifier, n, scoring, cv=None):

    # if no cv sets provided, split data in train and test sets
    if(cv is None):
        cv = cross_validation.StratifiedKFold(target, n)

    # return cross-validation scores
    return cross_validation.cross_val_score(classifier, data, target, cv=cv,
                                            scoring=scoring)


def grid_search(data, target, classifier, n, scoring, param, cv=None, cpu=1,
                log_f=None):
    '''
    This method does a CV grid search to find the best classifier parameters
    for the given data. The method returns the average CV-performance of the
    best parameters, and the best parameters.

    NOTE: data is assumed to be already scaled properly!

    data:       feature matrix
    target:     target class labels
    classifier: scikit-learn classifier object (with parameters set)
    param:      grid parameters that are suitable for the given classifier
    n:          number of cross-validation folds
    scoring:    scoring function to use as classifier performance measure
    cpu:        number of cpu's to use (didn't work for me thus far)
    log_f:      (open) file to log data to
    '''

    # if no cv sets provided, split data in train and test sets
    if(cv is None):
        cv = cross_validation.StratifiedKFold(target, n)

    # run grid search
    clf = GridSearchCV(classifier, param, scoring=scoring, cv=cv, n_jobs=cpu)
    clf.fit(data, target)

    # log results if requested
    if(log_f):
        for params, mean_score, scores in clf.cv_scores_:
            log_f.write('%0.3f;%0.3f;[%s];%r\n' % (mean_score, scores.std(),
                        ', '.join(['%.3f' % (s) for s in scores]), params))
        log_f.write('\n')

    # return best parameters, and score
    return (clf.best_score_, clf.best_params_)

#
# Methods for unscaled data
#


def cv_score(data, target, classifier, n, scoring, param=None, cv=None, cpu=1,
             log_f=None, standardize=True, return_trained_cl=True):
    '''
    A grid search is done if parameters (param) are provided. Otherwise the
    parameters in the provided classifier are used.
    '''

    # create stratified train and test set generator
    if(cv is None):
        cv = cross_validation.StratifiedKFold(target, n)

    cv_scores = []
    cv_params = []
    cv_confusion = []
    cv_all_scores = []
    cv_roc_curves = roc.RocCollection()
    predictions = []

    if(standardize):
        # create scaler and scale the data with it
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)

    print
    print 'start cross-validation...'
    print

    # outer CV
    for fold_i, (trn_indices, tst_indices) in enumerate(cv):

        # slice out the train and test data
        trn_data = data[trn_indices, :]
        tst_data = data[tst_indices, :]
        trn_target = target[trn_indices]
        tst_target = target[tst_indices]

        '''
        if(standardize):

            # create scaler using the training instances
            scaler = preprocessing.StandardScaler().fit(trn_data)

            # scale the train and test data
            trn_data = scaler.transform(trn_data)
            tst_data = scaler.transform(tst_data)
        '''

        # obtain the original classifier parameters
        classifier_param = classifier.get_params()

        # perform grid search, if parameters are provided
        if(param):

            if(log_f):
                log_f.write('CV-loop %i\n' % (fold_i))

            # optimize parameters on train set (s is train score)
            (s, p) = grid_search(trn_data, trn_target, classifier, n,
                                 scoring, param, cpu=cpu, log_f=log_f)

            # update parameters with the optimized ones
            classifier_param.update(p)

        # use parameters to create new classifier object and train it
        best_cl = type(classifier)(**classifier_param)
        best_cl.fit(trn_data, trn_target)

        # test the classifier on the test set
        (score, all_scores, confusion, roc_curve, probas) = test_classifier(
            tst_data, tst_target, best_cl, scoring)

        print 'fold %i: %.3f' % (fold_i, score)
        sys.stdout.flush()

        # store test scores for this cv loop
        cv_scores.append(score)
        cv_all_scores.append(all_scores)
        cv_confusion.append(confusion)
        if(roc_curve):
            cv_roc_curves.add(roc_curve)
        predictions.extend(zip(tst_indices, probas, tst_target))

        # store classifier parameters
        cv_params.append(classifier_param)

    print
    print 'cross-validation result: %.3f' % (numpy.mean(cv_scores))
    print
    sys.stdout.flush()

    # train classifier on full data set if requested
    all_data_cl = None
    if(return_trained_cl):

        '''
        # scale the whole data set
        if(standardize):

            # create scaler for full data set
            scaler = preprocessing.StandardScaler().fit(data)
            # scale data set
            data = scaler.transform(data)
        '''

        # obtain the original classifier parameters
        classifier_param = classifier.get_params()

        # perform grid search, if parameters are provided
        if(param):

            # optimize parameters on train set (s is train score)
            (s, p) = grid_search(data, target, classifier, n, scoring,
                                 param, cpu=cpu, log_f=log_f)

            # update parameters with the optimized ones
            classifier_param.update(p)

        # use parameters to create new classifier object and train it
        all_data_cl = type(classifier)(**classifier_param)
        all_data_cl.fit(data, target)

    # return average score over the cv loops
    return (cv_scores, cv_params, cv_confusion, cv_all_scores, cv_roc_curves,
            predictions, all_data_cl)


def ffs(data, target, classifier, n, scoring, param=None, cv=None,
        feat_names=None, log_f=None, standardize=True, cpu=1):
    '''
    Forward feature selection. Grid search that includes parameters and all
    possible combinations of features is to extensive. This method limits
    the amount of explored feature combinations.
    '''
    #TODO add all_data_cl

    if(feat_names):
        assert(data.shape[1] == len(feat_names))

    # Create stratified train and test set generator
    if(cv is None):
        cv = cross_validation.StratifiedKFold(target, n)

    # keep track of selected features [(tst_score, param, [feat_i])]
    cv_scores = []
    cv_params = []
    cv_confusion = []
    cv_all_scores = []
    cv_roc_curves = roc.RocCollection()
    predictions = []

    cv_featis = []

    (rand_score, max_score) = metric_rand_max_score[scoring]
    
    if(standardize):
        # create scaler and scale the data with it
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)


    # outer CV
    for fold_i, (trn_indices, tst_indices) in enumerate(cv):

        # slice out the train and test data
        trn_data = data[trn_indices, :]
        tst_data = data[tst_indices, :]
        trn_target = target[trn_indices]
        tst_target = target[tst_indices]

        '''
        if(standardize):
            # create scaler using the training instances
            scaler = preprocessing.StandardScaler().fit(trn_data)

            # scale the train and test data
            trn_data = scaler.transform(trn_data)
            tst_data = scaler.transform(tst_data)
        '''

        # initialize list of selected features
        select = [(rand_score, None, []),
                  (rand_score, None, [])]

        print
        print 'FEATURE SELECTION CV-LOOP %i' % (fold_i)
        print

        # keep selecting new features as long as:
        # - the score does not decrease
        # - max score has not been reached
        # - there are more features left
        #while((select[-1][0] >= select[-2][0]
        #        and select[-1][0] < max_score)
        #        and len(select) - 2 < data.shape[1]):
        for selection_i in xrange(data.shape[1]):

            # store results for each added feature of this loop
            results = []

            for feat_i in xrange(data.shape[1]):

                sys.stdout.write('.')
                if((feat_i + 1) % 80 == 0):
                    sys.stdout.write('\n')
                sys.stdout.flush()

                # only try features that are not already selected
                if not(feat_i in select[-1][2]):

                    # add current feature index to selection (copy)
                    feat_is = select[-1][2][:]
                    feat_is.append(feat_i)

                    # slice selected features from data
                    trn_data_part = trn_data[:, feat_is]

                    if(param):

                        # log to grid search file
                        if(log_f):
                            log_f.write('%s' % (str(feat_is)))
                            if(feat_names):
                                log_f.write('[%s]' % (', '.join([feat_names[i]
                                            for i in feat_is])))
                            log_f.write('\n')

                        # run parameter grid search
                        (best_s, best_p) = grid_search(
                            trn_data_part, trn_target, classifier, n, scoring,
                            param, log_f=log_f, cpu=cpu)
                    else:
                        # obtain cv score (grid search not neccasary)
                        best_p = classifier.get_params()
                        best_s = numpy.mean(cv_scores_no_scaling(
                            trn_data_part, trn_target, classifier, n, scoring))

                    # store the result
                    results.append((best_s, best_p, feat_is))

            # obtain the best score of this loop
            winner = sorted(results, key=operator.itemgetter(0))[-1]

            print('\nFeature %i: %s' % (len(select) - 1, str(winner)))

            select.append(winner)

        # pick the best model (the one before last in the selection)
        #if(len(select) == 3):
        #    (trn_score, bestp, feat_is) = select[2]
        #else:
        #    (trn_score, bestp, feat_is) = select[-2]
        # TODO: plot scores for the selection iterations...
        (trn_score, bestp, feat_is) = sorted(
            select, key=operator.itemgetter(0))[-1]

        # obtain original classifier parameters and update optimized ones
        classifier_param = classifier.get_params()
        if(bestp):
            classifier_param.update(bestp)

        # use these to create new classifier object and train it
        best_cl = type(classifier)(**classifier_param)
        best_cl.fit(trn_data[:, feat_is], trn_target)

        # slice selected features from test data
        tst_data = tst_data[:, feat_is]

        # test the classifier on the test set
        (score, all_scores, confusion, roc_curve, probas) = test_classifier(
            tst_data, tst_target, best_cl, scoring)

        # store test scores for this cv loop
        cv_scores.append(score)
        cv_all_scores.append(all_scores)
        cv_confusion.append(confusion)
        if(roc_curve):
            cv_roc_curves.add(roc_curve)
        predictions.extend(zip(tst_indices, probas, tst_target))

        # store classifier parameters
        cv_params.append(classifier_param)

        # store selected feature indices
        cv_featis.append(feat_is)

        print
        print 'fold %i: %.3f' % (fold_i, score)

    print
    print 'cross-validation result: %.3f' % (numpy.mean(cv_scores))
    print

    return (cv_scores, cv_params, cv_confusion, cv_all_scores, cv_roc_curves,
            cv_featis, predictions)


def bfs(data, target, classifier, n, scoring, param=None, cv=None,
        feat_names=None, log_f=None, standardize=True, cpu=1):
    '''
    Backward feature selection. Grid search that includes parameters and all
    possible combinations of features is to extensive. This method limits
    the amount of explored feature combinations.
    '''
    # TODO add all_data_cl

    if(feat_names):
        assert(data.shape[1] == len(feat_names))

    # Create stratified train and test set generator
    if(cv is None):
        cv = cross_validation.StratifiedKFold(target, n)

    # keep track of selected features [(tst_score, param, [feat_i])]
    cv_scores = []
    cv_params = []
    cv_confusion = []
    cv_all_scores = []
    cv_roc_curves = roc.RocCollection()
    predictions = []

    cv_featis = []

    (rand_score, max_score) = metric_rand_max_score[scoring]

    if(standardize):
        # create scaler and scale the data with it
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)

    # outer CV
    for fold_i, (trn_indices, tst_indices) in enumerate(cv):

        # slice out the train and test data
        trn_data = data[trn_indices, :]
        tst_data = data[tst_indices, :]
        trn_target = target[trn_indices]
        tst_target = target[tst_indices]

        '''
        if(standardize):
            # create scaler using the training instances
            scaler = preprocessing.StandardScaler().fit(trn_data)

            # scale the train and test data
            trn_data = scaler.transform(trn_data)
            tst_data = scaler.transform(tst_data)
        '''

        # keep track of removed features
        select = [(rand_score, None, []),
                  (rand_score, None, [])]

        print
        print 'FEATURE SELECTION CV-LOOP %i' % (fold_i)
        print

        # for now, let's test until we are at a single feature
        for selection_i in xrange(data.shape[1] - 1):

            # store results for each added feature of this loop
            results = []

            for feat_i in xrange(data.shape[1]):

                sys.stdout.write('.')
                if((feat_i + 1) % 80 == 0):
                    sys.stdout.write('\n')
                sys.stdout.flush()

                # only try features that are not already selected
                if not(feat_i in select[-1][2]):

                    # add current feature index to selection (copy)
                    remove_is = select[-1][2][:]
                    remove_is.append(feat_i)

                    feat_is = [fi for fi in xrange(data.shape[1])
                               if not fi in remove_is]

                    # slice selected features from data
                    trn_data_part = trn_data[:, feat_is]

                    if(param):

                        # log to grid search file
                        if(log_f):
                            log_f.write('%s' % (str(feat_is)))
                            if(feat_names):
                                log_f.write('[%s]' % (', '.join([feat_names[i]
                                            for i in feat_is])))
                            log_f.write('\n')

                        # run parameter grid search
                        (best_s, best_p) = grid_search(
                            trn_data_part, trn_target, classifier, n, scoring,
                            param, log_f=log_f, cpu=cpu)
                    else:
                        # obtain cv score (grid search not neccasary)
                        best_p = classifier.get_params()
                        best_s = numpy.mean(cv_scores_no_scaling(
                            trn_data_part, trn_target, classifier, n, scoring))

                    # store the result
                    results.append((best_s, best_p, remove_is))

            # obtain the best score of this loop
            winner = sorted(results, key=operator.itemgetter(0))[-1]

            print('\nFeature %i: %s' % (len(select) - 1, str(winner)))

            select.append(winner)

        # pick the best model (the one before last in the selection)
        #if(len(select) == 3):
        #    (trn_score, bestp, feat_is) = select[2]
        #else:
        #    (trn_score, bestp, feat_is) = select[-2]
        # TODO: plot scores for the selection iterations...
        (trn_score, bestp, remove_is) = sorted(
            select, key=operator.itemgetter(0))[-1]

        feat_is = [fi for fi in xrange(data.shape[1]) if not fi in remove_is]

        # obtain original classifier parameters and update optimized ones
        classifier_param = classifier.get_params()
        if(bestp):
            classifier_param.update(bestp)

        # use these to create new classifier object and train it
        best_cl = type(classifier)(**classifier_param)
        best_cl.fit(trn_data[:, feat_is], trn_target)

        # slice selected features from test data
        tst_data = tst_data[:, feat_is]

        # test the classifier on the test set
        (score, all_scores, confusion, roc_curve, probas) = test_classifier(
            tst_data, tst_target, best_cl, scoring)

        # store test scores for this cv loop
        cv_scores.append(score)
        cv_all_scores.append(all_scores)
        cv_confusion.append(confusion)
        if(roc_curve):
            cv_roc_curves.add(roc_curve)
        predictions.extend(zip(tst_indices, probas, tst_target))

        # store classifier parameters
        cv_params.append(classifier_param)

        # store selected feature indices
        cv_featis.append(feat_is)

        print
        print 'fold %i: %.3f' % (fold_i, score)

    print
    print 'cross-validation result: %.3f' % (numpy.mean(cv_scores))
    print

    return (cv_scores, cv_params, cv_confusion, cv_all_scores, cv_roc_curves,
            cv_featis, predictions)


def classify(data, classifier):

    # prediction class labels on data set
    pred = classifier.predict(data)

    # and predict probabilities (if possible)
    if(hasattr(classifier, 'predict_proba')):
        proba = classifier.predict_proba(data)
        # get the probabilities of class one
        # TODO this only works for 2-class problems...
        proba = proba[:, 1]
    elif(hasattr(classifier, 'decision_function')):
        proba = classifier.decision_function(data)
    else:
        proba = pred

    return (pred, proba)


def test_classifier(tst_data, tst_target, classifier, scoring):

    '''
    # prediction class labels on test set
    tst_pred = classifier.predict(tst_data)

    # and predict probabilities on test set (if possible)
    if(hasattr(classifier, 'predict_proba')):
        tst_proba = classifier.predict_proba(tst_data)
        # get the probabilities of class one
        # TODO this only works for 2-class problems...
        tst_proba = tst_proba[:, 1]
    elif(hasattr(classifier, 'decision_function')):
        tst_proba = classifier.decision_function(tst_data)
    else:
        tst_proba = tst_pred
    '''
    tst_pred, tst_proba = classify(tst_data, classifier)

    args_pred = [tst_target, tst_pred]
    args_proba = [tst_target, tst_proba]

    # get score
    if(scoring in all_score_input['proba']):
        score = all_score_funcs[scoring](*args_proba)
    else:
        score = all_score_funcs[scoring](*args_pred)

    # some score functions only work for binary case, in that case add -1
    # for a multiclass classifier
    all_scores = []
    for sn in all_score_names:
        sf = all_score_funcs[sn]
        try:
            if(sn in all_score_input['proba']):
                all_scores.append(sf(*args_proba))
            else:
                all_scores.append(sf(*args_pred))
        except Exception:
            all_scores.append(-1.0)

    # obtain confusion matrix
    confusion = metrics.confusion_matrix(tst_target, tst_pred)

    roc_curve = None
    if(len(set(tst_target)) == 2):
        # creat ROC curve
        roc_curve = roc.ROC(*args_proba, class0=0, class1=1)

    # TODO tst_probas do not are not always probabilities... maybe return both
    # raw predictions, probas, and decision_function?
    return(score, all_scores, confusion, roc_curve, tst_proba)


def lda_weights(data, target):

    # scale data
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)

    # train lda classifier
    clf = lda.LDA()
    clf.fit(data, target)

    return clf.coef_

#
# helper methods
#


def parse_feature_file(feature_f):
    '''
    Contains list of features to be tested per line in file. The first word
    is the name of the experiment.
    '''
    feature_experiments = []
    with open(feature_f, 'r') as fin:
        for line in fin:
            tokens = line.split()
            feature_experiments.append((tokens[0], tokens[1:]))
    return feature_experiments


def get_timed_parameter_range(classifier, data, target, standardize,
                              time_limit, param_name, param_range=None):

    if (param_range is None):
        param_range = default_param_range[param_name]
    else:
        assert(type(param_range) == list)
        assert(all([type(i) == int for i in param_range]))  # float?
        assert(all([i > 0 for i in param_range]))
        assert(len(param_range) == len(set(param_range)))

    # scale data
    if(standardize):
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)

    param_index = 0
    fast = True

    start_time = int(time.time())

    def test_cl(cl, data, target):
        cl.fit(data, target)
        if(hasattr(cl, 'predict_proba')):
            cl.predict_proba(data)
        elif(hasattr(cl, 'decision_function')):
            cl.decision_function(data)
        else:
            cl.predict(data)

    # wrap the fit function in a timed thread and start running
    timed_fit = timeout.add_timeout(test_cl, limit=time_limit)

    while param_index < len(param_range) and fast:

        # get classifier parameters
        classifier_param = classifier.get_params()
        classifier_param[param_name] = param_range[param_index]

        # create classifier
        cl = type(classifier)(**classifier_param)
        print cl.get_params()

        timed_fit(cl, data, target)

        sleep_time = 0
        while(not timed_fit.ready and sleep_time < time_limit):
            time.sleep(1)
            sleep_time += 1

        # check if he's done within time
        if(timed_fit.ready):
            param_index += 1
        else:
            fast = False

    run_time = int(time.time()) - start_time
    print('\nRUNTIME TIMING TEST: %i' % (run_time))

    return (param_range[:param_index], run_time)


def get_classifier(classifier_str):
    '''
    This functions maps the classifier string classifier_str to the
    corresponding classifier object with the default paramers set.
    '''

    # SVC
    if(classifier_str == 'linearsvc'):
        cl = svm.LinearSVC(**svm_default_param)
    elif(classifier_str == 'svc_linear'):
        libsvm_default_param['kernel'] = 'linear'
        cl = svm.SVC(**libsvm_default_param)
    elif(classifier_str == 'svc_rbf'):
        libsvm_default_param['kernel'] = 'rbf'
        cl = svm.SVC(**libsvm_default_param)
    # polynomial, sigmoid kernel
    # nuSVC
    # Nearest Neighbors (euclidian distance used by default)
    elif(classifier_str == 'kn_uniform'):
        kn_default_param['weights'] = 'uniform'
        cl = neighbors.KNeighborsClassifier(**kn_default_param)
    elif(classifier_str == 'kn_distance'):
        kn_default_param['weights'] = 'distance'
        cl = neighbors.KNeighborsClassifier(**kn_default_param)
    elif(classifier_str == 'rn_uniform'):
        rn_default_param['weights'] = 'uniform'
        cl = neighbors.RadiusNeighborsClassifier(**rn_default_param)
    elif(classifier_str == 'rn_distance'):
        rn_default_param['weights'] = 'distance'
        cl = neighbors.RadiusNeighborsClassifier(**rn_default_param)
    elif(classifier_str == 'nc'):
        cl = neighbors.NearestCentroid()
    # LDA and QDA, priors are by default set to 1/len(class) for each class
    elif(classifier_str == 'lda'):
        cl = lda.LDA()
    elif(classifier_str == 'qda'):
        cl = qda.QDA()
    # Gaussion naive bayes
    # from the code it is unclear how priors are set
    elif(classifier_str == 'gnb'):
        cl = naive_bayes.GaussianNB()
    elif(classifier_str == 'mnb'):
        cl = naive_bayes.MultinomialNB()
    elif(classifier_str == 'bnb'):
        cl = naive_bayes.BernoulliNB()
    # Decision tree
    elif(classifier_str == 'dtree'):
        cl = tree.DecisionTreeClassifier()
    elif(classifier_str == 'rforest'):
        cl = ensemble.RandomForestClassifier()
    else:
        # raise error if classifier not found
        raise ValueError('Classifier not implemented: %s' % (classifier_str))

    return (cl)


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--feature_matrix_dir', required=True)
    parser.add_argument('-l', '--labeling', required=True)
    parser.add_argument('-c', '--classifier', nargs='+', required=True)
    parser.add_argument('-n', '--n_fold_cv', type=int, required=True)
    parser.add_argument('-s', '--feature_selection', required=True)
    parser.add_argument('-e', '--evaluation_score', required=True)

    # root output directory
    parser.add_argument('-o', '--output_dir', required=True)

    parser.add_argument('--standardize', action='store_true', default=False)
    parser.add_argument('--classes', nargs='+', default=None)
    parser.add_argument('--features', nargs='+', default=None)
    parser.add_argument('--feature_file')
    parser.add_argument('--cross_validation_file')

    #parser.add_argument('--lda_weights', action='store_true', default=False)

    # parameter optimization?
    parser.add_argument('--timeout', type=int)  # seconds

    # parameter choices, disregarded if timeout is set
    parser.add_argument('--radius', nargs='+', default=None)
    parser.add_argument('--neighbors', nargs='+', default=None)
    parser.add_argument('--c_parameter', nargs='+', default=None)
    parser.add_argument('--gamma', nargs='+', default=None)

    parser.add_argument('--cpu', type=int, default=1)

    args = parser.parse_args()

    ###########################################################################
    # STEP 1: read feature file and cross-validation file
    ###########################################################################

    # read feature file, if provided
    if(args.feature_file):
        feature_experiments = parse_feature_file(args.feature_file)
    else:
        feature_experiments = [('exp', args.features)]

    # read cross-validation file, if provided
    if(args.cross_validation_file):
        cv = file_io.read_cross_validation(args.cross_validation_file)
    else:
        cv = None

    ###########################################################################
    # STEP 2: obtain performance score to use
    ###########################################################################

    # determine what performance score function to use
    if(args.evaluation_score not in all_score_names):
        print('\nIncorrect evaluation score: %s' % (args.evaluation_score))
        print('\nOptions: %s\n' % (', '.join(all_score_names)))
        sys.exit()
    scoring = args.evaluation_score

    ###########################################################################
    # STEP 3: obtain user provided classification parameters
    ###########################################################################

    # store user define parameters
    user_params = {}
    if(args.c_parameter):
        user_params['C'] = [float(p) for p in args.c_parameter]
    if(args.gamma):
        user_params['gamma'] = [float(p) for p in args.gamma]
    if(args.neighbors):
        user_params['n_neighbors'] = [int(p) for p in args.neighbors]
    if(args.radius):
        user_params['radius'] = [float(p) for p in args.radius]

    ###########################################################################
    # STEP 4: load the feature matrix
    ###########################################################################

    # load feature matrix
    print '\nLoading feature matrix...'
    fm = featmat.FeatureMatrix.load_from_dir(args.feature_matrix_dir)
    print 'Done.\n'

    ###########################################################################
    # STEP 5: check number of classes and compatibility with performance score
    ###########################################################################

    # determine how many classes there are
    if(args.classes):
        num_classes = len(args.classes)
    else:
        num_classes = len(fm.labeling_dict[args.labeling].class_names)

    # check if more than one class is provided
    if(num_classes < 2):
        print('\nProvide two or more classes.\n')
        sys.exit()

    # check if evaluation score is possible for given number of classes
    elif(num_classes > 2 and args.evaluation_score == 'roc_auc'):
        print('\nroc_auc only implemented for two class problems.\n')
        sys.exit()

    ###########################################################################
    # STEP 6: create output directory
    ###########################################################################

    # create result dir
    if not(os.path.exists(args.output_dir)):
        os.mkdir(args.output_dir)

    '''
    if(args.lda_weights):

        # for each feature set experiment
        for exp_name, feature_list in feature_experiments:

            # obtain scikit-learn dataset (NOTE not standardized)
            ds = fm.get_sklearn_dataset(feat_ids=feature_list,
                                        labeling_name=args.labeling,
                                        class_ids=args.classes,
                                        standardized=False)

            # obtain data and target from it
            data = ds.data
            target = ds.target

            # two-class only!!!
            weights = lda_weights(data, target)
            out_f = os.path.join(args.output_dir, 'lda_weights.txt')
            file_io.write_tuple_list(out_f, zip(feature_list, weights[:, 0]))

        sys.exit()
    '''

    # track runtime
    overall_start_time = int(time.time())

    ###########################################################################
    # LOOP outer: iterate over desired classifiers
    ###########################################################################

    # for each provided classifier
    for classifier_str in args.classifier:

        # create classifier output dir
        cl_d = os.path.join(args.output_dir, classifier_str)
        if not(os.path.exists(cl_d)):
            os.mkdir(cl_d)

        #######################################################################
        # LOOP inner: iterate over desired feature sets
        #######################################################################

        # for each feature set experiment
        for exp_name, feature_list in feature_experiments:

            # create output dir for this experiment
            exp_d = os.path.join(cl_d, exp_name)
            if not(os.path.exists(exp_d)):
                os.mkdir(exp_d)

            ###################################################################
            # Fetch classifier object and scikit-learn data set
            ###################################################################

            # obtain classifier with default parameters set
            cl = get_classifier(classifier_str)

            # obtain scikit-learn dataset
            # NOTE: feature matrix is not standardized)
            # NOTE: if feature_list is None, all features are used
            # NOTE: if args.classes is None, all classes are used
            ds = fm.get_sklearn_dataset(feat_ids=feature_list,
                                        labeling_name=args.labeling,
                                        class_ids=args.classes,
                                        standardized=False)

            # obtain data and target from it
            data = ds.data
            target = ds.target

            ###################################################################
            # Determine the classifier parameter(s/ ranges)
            ###################################################################

            # parameters dictionary
            param = {}

            # iterate over all possible classifier parameters
            for par in classifier_params:

                # check if the current classifier uses this parameter
                if(classifier_str in classifiers_per_param[par]):

                    # use user defined one, if provided
                    if(par in user_params.keys()):
                        param[par] = user_params[par]

                    # use default range otherwise
                    else:
                        param[par] = default_param_range[par]

                        ''' remove timeout for the moment
                        # adjust range if timeout is provided
                        if(args.timeout and par in timed_param):
                            param[par], run_time = get_timed_parameter_range(
                                cl, data, target, args.standardize,
                                args.timeout, par)

                            # check parameter range
                            if(len(param[par]) == 0):
                                print('Time out occured.\n')
                                sys.exit()
                            elif(len(param[par]) == 1):
                                tmp_param = {par: param[par][0]}
                                cl.set_params(**tmp_param)
                                del param[par]
                            else:
                                pass
                        '''

                    # set parameter
                    if(len(param[par]) == 0):  # only in case of timeout???
                        print 'No value for parameter: %s' % (par)
                        sys.exit(1)
                    elif(len(param[par]) == 1):
                        # set parameter, no grid search required
                        tmp_param = {par: param[par][0]}
                        cl.set_params(**tmp_param)
                        del param[par]
                    else:
                        # otherwise keep in param dict, used to run grid search
                        pass

            ''' remove time estimate for now
            print cl.get_params()
            if(param and args.timeout):
                print param
                # estimate time
                time_estimate = run_time * args.n_fold_cv
                if(classifier_str in classifiers_per_param['gamma']):
                    if 'gamma' in param:
                        time_estimate *= len(param['gamma'])
                print('Estimated run time (sec): %i' % (time_estimate))
            '''

            ###################################################################
            # define output files
            ###################################################################

            settings_f = os.path.join(exp_d, 'settings.txt')
            result_f = os.path.join(exp_d, 'result.txt')
            cm_f = os.path.join(exp_d, 'confusion_matrix.txt')
            gs_f = os.path.join(exp_d, 'grid_search.txt')
            fs_f = os.path.join(exp_d, 'feature_selection.txt')
            param_f = os.path.join(exp_d, 'parameters.txt')
            roc_f = os.path.join(exp_d, 'roc.txt')
            roc_fig_f = os.path.join(exp_d, 'roc.png')
            predictions_f = os.path.join(exp_d, 'predictions.txt')
            all_data_cl_f = os.path.join(exp_d, 'classifier.joblib.pkl')

            ###################################################################
            # RUN EXPERIMENT
            # - cv_score (feature selection 'none')
            # - ffs
            # - bfs
            ###################################################################

            print args.feature_selection

            # catch warnings from lda and qda as exepctions
            warnings.filterwarnings(action='error',
                                    category=RuntimeWarning)

            gs_log_f = open(gs_f, 'w')
            gs_log_f.write('mean,std,cv_scores,parameters\n\n')

            cv_roc_curves = None

            try:

                # run CV experiment without feature selection
                if(args.feature_selection == 'none'):
                    print 'start cv score...'
                    (cv_scores, cv_params, cv_confusion, cv_all_scores,
                        cv_roc_curves, predictions, all_data_cl) = cv_score(
                            data, target, cl, args.n_fold_cv, scoring,
                            param=param, cv=cv, log_f=gs_log_f, cpu=args.cpu,
                            standardize=args.standardize)
                    cv_feat_is = None

                # run CV experiment with forward feature selection
                # TODO all_data_cl
                elif(args.feature_selection == 'ffs'):
                    print 'start ffs...'
                    (cv_scores, cv_params, cv_confusion, cv_all_scores,
                        cv_roc_curves, cv_feat_is, predictions) =\
                        ffs(data, target, cl, args.n_fold_cv, scoring,
                            param=param, cv=cv, log_f=gs_log_f, cpu=args.cpu,
                            standardize=args.standardize)

                # run CV experiment with backward feature selection
                # TODO all_data_cl
                elif(args.feature_selection == 'bfs'):
                    print 'start bfs...'
                    (cv_scores, cv_params, cv_confusion, cv_all_scores,
                        cv_roc_curves, cv_feat_is, predictions) =\
                        bfs(data, target, cl, args.n_fold_cv, scoring,
                            param=param, cv=cv, log_f=gs_log_f, cpu=args.cpu,
                            standardize=args.standardize)

                else:
                    cv_scores = 'Feature selection method does not exist.'

            except(RuntimeWarning) as e:
                #cv_scores = 'RuntimeWarning occured: %s' % rw
                print traceback.format_exc()
                raise e
                sys.exit()
            except Exception as e:
                print traceback.format_exc()
                raise e
                sys.exit()
            finally:
                gs_log_f.close()

            ###################################################################
            # Write experiment results
            ###################################################################

            # write results to output files
            if(type(cv_scores) == str):
                # a bit of a hack to write error to file
                with open(result_f, 'w') as fout:
                    fout.write(cv_scores)
            else:

                # write settings to file, needs to be improved
                # TODO turn into function
                with open(settings_f, 'w') as fout:
                    fout.write('sample_names,feature_names,target_names,' +
                               'classifier_name,classifier_params,' +
                               'grid_params,n_fold_cv,feature_selection\n')
                    first_sample_names = ds['sample_names'][:10]
                    first_sample_names.append('...')
                    fout.write('%s\n' % (str(first_sample_names)))
                    fout.write('%s\n' % (str(ds['feature_names'])))
                    fout.write('%s\n' % (str(ds['target_names'])))
                    fout.write('%s\n' % (str(classifier_str)))
                    fout.write('%s\n' % (str(cl.get_params())))
                    fout.write('%s\n' % (str(param).replace('\n', '')))
                    fout.write('%i\n' % (args.n_fold_cv))
                    fout.write('%s\n' % (args.feature_selection))

                # write the cv performance results
                with open(result_f, 'w') as fout:
                    fout.write('%s\n' % (','.join(all_score_names)))
                    for index in range(len(all_score_names)):
                        s = [item[index] for item in cv_all_scores]
                        fout.write('%s\n' % (str(s)))

                # write confusion matrices
                with open(cm_f, 'w') as fout:
                    for index, cm in enumerate(cv_confusion):
                        fout.write('CV%i\n' % (index))
                        fout.write('%s\n\n' % (str(cm)))

                # write parameters
                with open(param_f, 'w') as fout:
                    for cv_param in cv_params:
                        fout.write('%s\n' % (str(cv_param)))

                # plot roc curves
                if not(cv_roc_curves.is_empty()):
                    cv_roc_curves.save_avg_roc_plot(roc_fig_f)

                # store classifier trained on full data set
                if not(all_data_cl is None):
                    _ = joblib.dump(all_data_cl, all_data_cl_f, compress=9)

                # sort predictions by object index
                sorted_predictions = sorted(predictions,
                                            key=operator.itemgetter(0))
                preds = zip(fm.object_ids, [p[1] for p in sorted_predictions],
                            [p[2] for p in sorted_predictions])
                file_io.write_tuple_list(predictions_f, preds)

                # write feature selection
                if(cv_feat_is):

                    with open(fs_f, 'w') as fout:
                        fout.write('cv_loop,selected features\n')
                        for index, fs in enumerate(cv_feat_is):
                            fout.write('%i,%s\n' % (index,
                                       '\t'.join([feature_list[fi]
                                       for fi in fs])))

    print('\nRUNTIME: %i' % (int(time.time() - overall_start_time)))
