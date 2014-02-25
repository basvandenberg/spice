import os
import sys
import operator

import numpy

# HACK TODO remove if sklearn is updated to 0.14 on compute servers...
import sklearn
if not(sklearn.__version__ == '0.14.1'):
    sys.path.insert(1, os.environ['SKL'])
    reload(sklearn)
assert(sklearn.__version__ == '0.14.1')

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

from biopy import roc


# classification performance measures
all_score_names = ['roc_auc', 'mcc', 'f1', 'precision', 'average_precision',
                   'recall', 'accuracy']
all_score_funcs = dict(zip(all_score_names, [
    metrics.roc_auc_score,
    metrics.matthews_corrcoef,
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
    'C': 10.0 ** numpy.arange(-3, 4),
    'gamma': 10.0 ** numpy.arange(-1, 2)
}

# timed parameters
# timed_param = ['C', 'radius', 'n_neighbors']

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
    clf = GridSearchCV(classifier, param, scoring=scoring, cv=cv, refit=False,
                       n_jobs=cpu)
    clf.fit(data, target)

    # log results if requested
    if(log_f):
        for params, mean_score, scores in clf.grid_scores_:
            log_f.write('%0.3f;%0.3f;[%s];%r\n' % (mean_score, scores.std(),
                        ', '.join(['%.3f' % (s) for s in scores]), params))
        log_f.write('\n')

    # return best parameters, and score
    return (clf.best_score_, clf.best_params_)

#
# Methods for unscaled data
#


def cv_score(data, target, classifier, n, scoring, param=None, cv=None, cpu=1,
             log_f=None, standardize=True, refit=True):
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
            _, p = grid_search(trn_data, trn_target, classifier, n,
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
    if(refit):

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
    # TODO check if -1 should be changed... -1 is valid output for MCC score, 
    #      which only is possible for binary cases.
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

'''
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
'''


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


#if __name__ == '__main__':
# TODO add test run
