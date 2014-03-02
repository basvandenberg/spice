import os
import operator
import glob
import datetime
import time
import numpy
import shutil
import traceback
#import urllib2
#import random

from spice import featext
from spice import featmat
from biopy import sequtil
from biopy import file_io


class ProjectManager(object):

    TIMEOUT = 20  # sec

    def __init__(self, root_dir, ref_data_dir):
        self.root_dir = root_dir
        self.ref_data_dir = ref_data_dir
        self.user_id = None
        self.project_id = None

    def set_user(self, user_id):
        '''
        Sets the current user. Needs to be set before the other functions can
        be used. NOT CHECKED OR ANYTHING
        '''

        # store user id
        self.user_id = user_id

        # set path to user dir
        self.user_dir = os.path.join(self.root_dir, self.user_id)

    def set_project(self, project_id):
        '''
        Sets the current project to project_id and sets project paths
        '''

        # store project id
        self.project_id = project_id

        if not(self.project_id is None):

            # set path to project dir
            self.project_dir = os.path.join(self.user_dir, self.project_id)

            # set paths to feature extraction and classification dir
            self.fe_dir = os.path.join(self.project_dir, 'feature_extraction')
            self.fm_dir = os.path.join(self.fe_dir, 'feature_matrix_protein')
            self.cl_dir = os.path.join(self.project_dir, 'classification')

            # set paths to jobs dir, and job status sub-directories
            self.job_dir = os.path.join(self.project_dir, 'jobs')
            self.job_running_dir = os.path.join(self.job_dir, 'running')
            self.job_done_dir = os.path.join(self.job_dir, 'done')
            self.job_error_dir = os.path.join(self.job_dir, 'error')
            self.job_waiting_dir = os.path.join(self.job_dir, 'waiting')

            # path to project details file
            self.project_details_f = os.path.join(self.project_dir,
                                                  'project_details.txt')

            # paths to data files
            self.object_ids_f = os.path.join(self.project_dir, 'ids.txt')
            self.labels_f = os.path.join(self.project_dir, 'labels.txt')
            self.protein_seqs_f = os.path.join(self.project_dir,
                                               'proteins.fsa')
            self.orf_seqs_f = os.path.join(self.project_dir, 'orfs.fsa')
            self.sec_struct_f = os.path.join(self.project_dir,
                                             'sec_struct.fsa')
            self.solv_access_f = os.path.join(self.project_dir,
                                              'solv_access.fsa')
            self.hist_dir = os.path.join(self.fe_dir, 'histograms')
            self.fm_f = os.path.join(self.fm_dir, 'feat.mat')
            self.feat_ids_f = os.path.join(self.fe_dir, 'feat_ids.txt')

    def get_projects(self):
        '''
        Returns all project ids and initiation times for the current user.
        '''
        project_ids = []
        if not(self.user_id is None):
            for d in glob.glob(os.path.join(self.user_dir, '*')):
                if(os.path.isdir(d)):
                    f = os.path.join(d, 'project_details.txt')
                    with open(f, 'r') as fin:
                        name = fin.readline().split()[1]
                        init = fin.readline().split()[1]
                    project_ids.append((name, init))
            project_ids = sorted(project_ids, reverse=True)
        return project_ids

    def get_feature_extraction(self):
        '''
        Returns feature extraction object of the project with project_id for
        the user with user_id.
        pre: user_id is set
        pre: project_id is set
        '''
        fe = None
        if not(self.project_id is None):
            fe = featext.FeatureExtraction()
            fe.set_root_dir(self.fe_dir)
            fe.load()
        return fe

    def get_feature_matrix(self):
        '''
        Returns feature matrix object of the project with project_id for
        the user with user_id.
        '''
        fm = None
        if not(self.project_id is None):
            fm = featmat.FeatureMatrix.load_from_dir(self.fm_dir)
        return fm

    # helper function
    def timestamp_str(self):
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

    ###########################################################################
    # Job management stuff, would be nice to use some readily available
    # module are anything for this...
    ###########################################################################

    def get_job_id(self):
        return self.timestamp_str()

    def fetch_job_files(self, app):
        '''
        Used by classification list page
        '''
        status_dirs = {
            'done': self.job_done_dir,
            'running': self.job_running_dir,
            'error': self.job_error_dir,
            'waiting': self.job_waiting_dir
        }

        app_files = []

        for status in status_dirs.keys():

            dir = status_dirs[status]
            files = [f for f in glob.glob(os.path.join(dir, '*'))]

            for f in files:
                a = self.parse_job_file(f)
                if(app == a):
                    app_files.append((os.path.basename(f), f, status))

        return sorted(app_files, key=operator.itemgetter(0), reverse=True)

    def get_feat_calc_status(self):
        '''
        This function returns a dictionary that maps a status to a list of jobs
        that have this status. The list of jobs is sorted by initiation time.
        '''
        cat_status = {}

        if not(self.project_id is None):

            status_dirs = {
                #'done': self.job_done_dir,
                'running': self.job_running_dir,
                'error': self.job_error_dir,
                'waiting': self.job_waiting_dir
            }

            for status in status_dirs.keys():

                # initialize empty list
                jobs_list = []

                for f in glob.glob(os.path.join(status_dirs[status], '*')):
                    jobs_list.extend(self.parse_featext_job_file(f))

                # sort by initiation date
                sortl = sorted(jobs_list, key=operator.itemgetter(1),
                               reverse=True)

                # transform datetime objects to strings
                format_str = '%d-%m-%Y / %H:%M:%S'
                tstr = [(c, datetime.datetime.strftime(t, format_str))
                        for c, t in sortl]

                cat_status[status] = tstr

        return cat_status

    def parse_featext_job_file(self, f):
        with open(f, 'r') as fin:

            datestr, timestr, _ = os.path.basename(f).split('_')
            t = datetime.datetime.strptime(datestr + timestr, '%Y%m%d%H%M%S')

            tokens = fin.readline().split()
            app = os.path.splitext(os.path.basename(tokens[0]))[0]

            result = []

            if(app == 'featext'):
                try:
                    seqfeat_i = tokens.index('--protein_features')
                    index = seqfeat_i + 1
                    while(index < len(tokens) and
                            not tokens[index][0] == '-'):
                        result.append((tokens[index], t))
                        index += 1
                except ValueError:
                    pass

            return result

    def parse_job_file(self, f):
        with open(f, 'r') as fin:
            cmd = fin.readline()
            app_path = cmd.split()[0]
            app = os.path.splitext(os.path.basename(app_path))[0]
        return app

    def parse_classify_job_files(self, cl_id):
        '''
        Returns statuses of projects that are being classified.
        '''
        cat_status = {}

        if not(self.project_id is None):

            status_dirs = {
                'done': self.job_done_dir,
                'running': self.job_running_dir,
                'error': self.job_error_dir,
                'waiting': self.job_waiting_dir
            }

            for status in status_dirs.keys():

                # initialize empty list
                data_set_list = []

                for f in glob.glob(os.path.join(status_dirs[status], '*')):
                    with open(f, 'r') as fin:
                        cmd = fin.readline()
                    tokens = cmd.split()
                    if(tokens[0] == 'classify'):
                        assert(tokens[1] == '-f')
                        assert(tokens[3] == '-c')
                        cid = os.path.basename(
                            os.path.dirname(
                            os.path.dirname(tokens[4])))

                        if(cid == cl_id):
                            data_set = os.path.basename(
                                os.path.dirname(
                                os.path.dirname(tokens[2])))
                            data_set_list.append(data_set)

                status_dirs[status] = data_set_list

        return status_dirs

    ###########################################################################
    # Functions to obtain classification data
    ###########################################################################

    def get_cl_dir(self, cl_id):
        ''' This function returns the results directory for classifier cl_id.
        '''

        # there should be only one dir... this is a bit of hack to get it
        cl_base_dir = os.path.join(self.cl_dir, cl_id)
        dirs = []
        for d in glob.glob('%s/*/*' % (cl_base_dir)):
            dirs.append(d)

        assert(len(dirs) < 2)

        if(len(dirs) == 0):
            return None
        else:
            return dirs[0]

    def get_roc_f(self, cl_id):
        roc_f = os.path.join(self.get_cl_dir(cl_id), 'roc.png')
        if(os.path.exists(roc_f)):
            return roc_f
        else:
            return None

    def get_prediction_f(self, cl_id):
        pred_f = os.path.join(self.get_cl_dir(cl_id), 'predictions.txt')
        return pred_f

    def get_classifier_f(self, cl_id):
        cl_f = os.path.join(self.get_cl_dir(cl_id), 'classifier.joblib.pkl')
        if(os.path.exists(cl_f)):
            return cl_f
        else:
            return None

    def get_classifier_progress(self, cl_id):

        f = os.path.join(self.cl_dir, cl_id, 'progress.txt')
        result = ''
        with open(f, 'r') as fin:
            for line in fin:
                result += line
        return result

    def get_classifier_error(self, cl_id):

        f = os.path.join(self.cl_dir, cl_id, 'error.txt')
        error = ''
        with open(f, 'r') as fin:
            for line in fin:
                error += line
        return error

    def get_classifier_finished(self, cl_id):
        cl_d = self.get_cl_dir(cl_id)
        if(cl_d):
            result_f = os.path.join(self.get_cl_dir(cl_id), 'result.txt')
            return os.path.exists(result_f) and os.path.getsize(result_f) > 0
        else:
            return False

    def get_classifier_settings(self, cl_id):
        settings_f = os.path.join(self.get_cl_dir(cl_id), 'settings.txt')
        return file_io.read_settings_dict(settings_f)

    def get_classifier_ids(self):

        cl_ids = []

        # iterate over all directories in the classification dir
        for d in glob.glob(os.path.join(self.cl_dir, '*')):
            if os.path.isdir(d):
                cl_id = os.path.basename(d)
                cl_ids.append(cl_id)

        return cl_ids

    def get_classifier_result(self, cl_id):
        '''
        Reads result.txt file, returns cv_scores per score metric.
        {score_name: [cv_scores, ...], ...}
        '''

        cv_results = {}
        avg_results = {}
        with open(os.path.join(self.get_cl_dir(cl_id), 'result.txt'), 'r')\
                as fin:
            # TODO use a file_io read function?
            score_names = fin.readline().strip().split(',')
            for key in score_names:
                cv_s = eval(fin.readline())
                cv_results[key] = cv_s
                avg_results[key] = (numpy.mean(cv_s), numpy.std(cv_s))

        return (cv_results, avg_results)

    def get_all_classifier_results(self):

        # store classification results (in a list) in a dictionary with
        # class label set as key
        cl_results = {}

        # iterate over all directories in the classification dir
        for cl_id in self.get_classifier_ids():

            # check if the classifier is finished
            if(self.get_classifier_finished(cl_id)):

                # obtain classifier settings and result
                cv_results, avg_results = self.get_classifier_result(cl_id)
                cl_settings = self.get_classifier_settings(cl_id)

                # create class_ids key
                key = tuple(sorted(cl_settings['target_names']))
                cl_results.setdefault(key, {})

                cl_dict = {'cl_settings': cl_settings,
                           'cv_results': cv_results,
                           'avg_results': avg_results}
                cl_results[key][cl_id] = cl_dict

        return cl_results

    ###########################################################################
    # Functions that check form input and forward it to data structures
    ###########################################################################

    def delete_project(self, project_id):
        '''
        Delete the project with project_id
        pre: user_id is set
        '''
        project_dir = os.path.join(self.user_dir, project_id)
        if(os.path.exists(project_dir)):
            shutil.rmtree(project_dir)

    # start a new project
    def start_new_project(self, project_id, fasta_file, sequence_type,
                          reference_taxon=None):
        '''
        TOCHECK: what is fasta_file for type ???
        pre: sequence_type is orf_seq or prot_seq
        pre: user_id is set
        '''

        self.set_project(project_id)

        # check if user allready has a dir and create one if needed
        if not os.path.exists(self.user_dir):
            os.mkdir(self.user_dir)

        # check if a project with the same name exists
        if os.path.exists(self.project_dir):
            return 'A project with the same project id allready exists'

        # download reference fasta file
        ref_seqs = []
        if not(reference_taxon is None):

            # TODO
            if(sequence_type == 'orf_seq'):
                return 'Reference set can only be compared to protein' +\
                       'amino acid sequences, not to ORF sequences.'

            # obtain reference proteome sequences
            ref_f = os.path.join(self.ref_data_dir,
                                 '%i.fsa' % (reference_taxon))
            ref_red_f = os.path.join(self.ref_data_dir,
                                     '%i_reduced.fsa' % (reference_taxon))
            # first check local dir
            if(os.path.exists(ref_red_f)):
                ref_seqs = [s for s in file_io.read_fasta(ref_red_f)]
            elif(os.path.exists(ref_f)):
                ref_seqs = [s for s in file_io.read_fasta(ref_f)]
            # otherwise fetch reference data set
            else:
                pass
                '''
                url = 'http://www.uniprot.org/uniref/' +\
                      '?query=uniprot:(organism:%i+' % (reference_taxon) +\
                      'keyword:181)+identity:0.5&format=fasta'
                response = urllib2.urlopen(url)
                try:
                    ref_seqs = [s for s in file_io.read_fasta(response)]
                except Exception:
                    return 'There appears to be an error in the reference ' +\
                           'data fasta file'

                # check if reference data set is not too large
                max_num_seqs = 15000
                if(len(ref_seqs) > max_num_seqs):
                    # randomly select 15000 sequences
                    indices = random.sample(range(len(ref_seqs)), max_num_seqs)
                    ref_seqs = [ref_seqs[i] for i in indices]
                '''

        # estimate reference data set size
        size = len(ref_seqs) * 285  # estimate 285 bytes per seq

        # check file size
        max_size = 5243000  # bytes (5MB)
        while True:
            data = fasta_file.file.read(8192)
            if(size > max_size):
                return 'Sequence data exeeds the maximum ' +\
                       'allowed size (5MB)'

            if not(data):
                break
            size += len(data)
        if(size > max_size):
            return 'Sequence data exeeds the maximum ' +\
                   'allowed size (5MB)'

        # reset to beginning of fasta file
        fasta_file.file.seek(0)

        # read sequences from fasta file (to obtain object ids...)
        try:
            seqs = [s for s in file_io.read_fasta(fasta_file.file)]
            seqs.extend(ref_seqs)
            ids = [s[0] for s in seqs]
        except Exception as e:
            return str(e) +\
                'Please consult the documentation (<i>file formats</i> ' +\
                'section) to learn more about the FASTA file format.'

        # reset pointer to begin of file
        #fasta_file.file.seek(0)
        # close the temporary file, not sure if this is neccesary
        fasta_file.file.close()

        # create sequence feature extraction object to check input
        fe = featext.FeatureExtraction()
        try:
            fe.set_protein_ids(ids)
            fe.protein_data_set.set_data_source(sequence_type, seqs)
            # translate to prot seq if orf provided
            if(sequence_type == 'orf_seq'):
                ids = [s[0] for s in seqs]
                prot_seqs = [(sequtil.translate(s[1])) for s in seqs]
                # chop off translated stop codons at terminus
                prot_seqs = [s[:-1] if s[-1] == '*' else s for s in prot_seqs]
                fe.protein_data_set.set_data_source('prot_seq',
                                                    zip(ids, prot_seqs))
        except ValueError as e:
            print(traceback.format_exc())
            return str(e)
        except:
            print(traceback.format_exc())
            return 'Error during initiation new project'

        # add labeling in case of added reference set
        if(len(ref_seqs) > 0):
            l = [(s[0], 0) for s in seqs]
            l.extend([(s[0], 1) for s in ref_seqs])
            class_names = ['dataset', 'taxon%i' % (reference_taxon)]
            fe.fm_protein.add_labeling('reference', dict(l), class_names)

        # create data directory for this project (just to be sure, check again)
        if not(os.path.exists(self.project_dir)):
            os.mkdir(self.project_dir)

            # and create directories to store job status
            os.mkdir(self.job_dir)
            os.mkdir(self.job_waiting_dir)
            os.mkdir(self.job_running_dir)
            os.mkdir(self.job_done_dir)
            os.mkdir(self.job_error_dir)

            # create classification dir
            os.mkdir(self.cl_dir)

        else:
            return 'A project with the same project id allready exists'

        # create project details file
        with open(self.project_details_f, 'w') as fout:
            fout.write('project_id\t%s\n' % (self.project_id))
            fout.write('project_init\t%s\n' % (self.timestamp_str()))

        # store feature extraction data
        fe.set_root_dir(self.fe_dir)
        fe.save()

        return ''

    # TODO naming is not so nice..., considering function above...
    # also to much of a duplicate of above function
    def start_example_project(self, project_id, fasta_f, seq_type, labeling_f):
        ''' Start new project without checking input data
        '''

        self.set_project(project_id)

        # check if user allready has a dir and create one if needed
        if not os.path.exists(self.user_dir):
            os.mkdir(self.user_dir)

        # check if a project with the same name exists, otherwise add number
        if(os.path.exists(self.project_dir)):
            index = 0
            while(os.path.exists(self.project_dir)):
                project_id = '%s_%i' % (self.project_id.split('_')[0], index)
                self.set_project(project_id)
                index += 1

        # read data from file
        try:
            seqs = [s for s in file_io.read_fasta(fasta_f)]
            ids = [s[0] for s in seqs]
        except Exception as e:
            print e
            return 'Error in fasta file'

        # create sequence feature extraction object
        fe = featext.FeatureExtraction()

        # set protein data
        try:
            fe.set_protein_ids(ids)
            fe.protein_data_set.set_data_source(seq_type, seqs)
            # translate to prot seq if orf provided
            if(seq_type == 'orf_seq'):
                ids = [s[0] for s in seqs]
                prot_seqs = [(sequtil.translate(s[1])) for s in seqs]
                # chop off translated stop codons at terminus
                prot_seqs = [s[:-1] if s[-1] == '*' else s for s in prot_seqs]
                fe.protein_data_set.set_data_source('prot_seq',
                                                    zip(ids, prot_seqs))
        except ValueError as e:
            print(traceback.format_exc())
            return str(e)
        except:
            print(traceback.format_exc())
            return 'Error during initiation new project'

        # add to feature matrix
        try:
            labeling_name = os.path.splitext(os.path.basename(labeling_f))[0]
            fe.fm_protein.add_labeling_from_file(labeling_name, labeling_f)
        except ValueError as e:
            return str(e)

        # create data directory for this project (just to be sure, check again)
        if not(os.path.exists(self.project_dir)):
            os.mkdir(self.project_dir)

            # and create directories to store job status
            os.mkdir(self.job_dir)
            os.mkdir(self.job_waiting_dir)
            os.mkdir(self.job_running_dir)
            os.mkdir(self.job_done_dir)
            os.mkdir(self.job_error_dir)

            # create classification dir
            os.mkdir(self.cl_dir)

        else:
            return 'A project with the same project id allready exists'

        # create project details file
        with open(self.project_details_f, 'w') as fout:
            fout.write('project_id\t%s\n' % (self.project_id))
            fout.write('project_init\t%s\n' % (self.timestamp_str()))

        # store feature extraction data
        fe.set_root_dir(self.fe_dir)
        fe.save()

        return ''

    def add_data_source(self, data_type, data_file, mapping_f=None):

        # for now, only fasta files are handled as data_path, no dirs or zips
        # with data and correspondig mapping_files.

        # for now, it is also required that there is a data item for each
        # object for which we have an id.

        # read sequences from fasta file
        try:
            seqs = [s for s in file_io.read_fasta(data_file)]
            ids = [s[0] for s in seqs]
        except Exception as e:
            # TODO
            print '\n%s\n%s\n%s\n' % (e, type(e), e.args)
            return 'Error in fasta file.'

        # close the temporary file, not sure if this is neccesary
        data_file.close()

        if not(len(set(ids)) == len(ids)):
            return 'Fasta file contains duplicate ids.'

        fe = self.get_feature_extraction()

        if not(set(fe.fm_protein.object_ids) == set(ids)):
            return 'Ids in provided file do not correspond to ids in project.'

        # reorder the sequence to the project object ids
        seq_dict = dict(seqs)
        seqs = [(sid, seq_dict[sid]) for sid in fe.fm_protein.object_ids]

        # create a uni_orf_mapping???

        try:
            fe.protein_data_set.set_data_source(data_type, seqs)
        except ValueError as e:
            print(traceback.format_exc())
            return str(e)

        # save feature extraction, if it all went well
        fe.save()

        return ''

    def add_labeling(self, labeling_name, labeling_f):

        # read labeling
        try:
            label_dict, class_names = file_io.read_labeling(labeling_f)
        except:
            return 'Error in labeling file.'
        finally:
            labeling_f.close()

        # add to feature matrix
        fm = self.get_feature_matrix()
        try:
            fm.add_labeling(labeling_name, label_dict, class_names)
        except ValueError as e:
            return str(e)

        # save if everything went well
        fm.save_to_dir(self.fm_dir)

        return ''

    # add custom features
    def add_custom_features(self, project_id, object_ids_f, feature_matrix_f):
        '''
        '''

        self.set_project(project_id)

        try:
            object_ids = [i for i in file_io.read_ids(object_ids_f.file)]
        except Exception as e:
            print '\n%s\n%s\n%s\n' % (e, type(e), e.args)
            return 'Error in object ids file'
        object_ids_f.file.close()

        try:
            featmat = numpy.loadtxt(feature_matrix_f.file)
        except Exception as e:
            print '\n%s\n%s\n%s\n' % (e, type(e), e.args)
            return 'Error in feature matrix file'
        feature_matrix_f.file.close()

        fm = self.get_feature_matrix()

        if not(sorted(object_ids) == sorted(fm.object_ids)):
            return 'The protein ids do not correspond to the proteins ' +\
                   'in this project'

        if not(featmat.shape[0] == len(fm.object_ids)):
            return 'The number of rows in the feature matrix does not ' +\
                   'correspond to the number of proteins in this project.'

        # reorder feature matrix rows
        featmat = featmat[fm.object_indices(object_ids)]

        try:
            fm.add_custom_features(featmat)
        except ValueError as e:
            return str(e)
        except Exception as e:
            print e
            return 'Something went wrong while adding custom features'

        fm.save_to_dir(self.fm_dir)

        return ''

    ###########################################################################
    # Functions that write a job file and add the job to the job queue (put it
    # in the waiting dir) These are the jobs which need to be runned using
    # SPiCE on the compute servers.
    ###########################################################################

    def run_feature_extraction(self, feature_categories):

        # obtain job id
        jobid = self.get_job_id()

        # build feature extraction (featext) command
        cmd = 'featext -r %s' % (self.fe_dir)
        cmd += ' --protein_features ' + ' '.join(feature_categories)

        # output files
        log_f = os.path.join(self.fe_dir, 'log.txt')
        error_f = os.path.join(self.fe_dir, 'error.txt')

        file_content = '%s\n%s\n%s\n' % (cmd, log_f, error_f)

        # check if the same job is already in one of the job directories
        found = False
        dirs = [self.job_waiting_dir, self.job_running_dir, self.job_done_dir]
        for d in dirs:
            for f in glob.glob(os.path.join(d, '*')):
                str = ''
                with open(f, 'r') as fin:
                    for line in fin:
                        str += line
                if(str == file_content):
                    found = True

        if not(found):
            # write the job to the queue
            with open(os.path.join(self.job_waiting_dir, jobid), 'w') as fout:
                fout.write(file_content)

    def run_classification(self, classifier, n_fold_cv, labeling_name,
                           class_ids, feat_ids, eval_score='roc_auc',
                           featsel=None):

        # obtain job id
        jobid = self.get_job_id()

        # create job dir
        out_dir = os.path.join(self.cl_dir, jobid)
        os.mkdir(out_dir)

        # output files
        progress_f = os.path.join(out_dir, 'progress.txt')
        error_f = os.path.join(out_dir, 'error.txt')

        # set feature selection to none if not providede
        if(featsel is None):
            featsel = 'none'

        # set evaluation score to f1 in case of multiple class classification
        class_ids = class_ids.split(',')
        if(len(class_ids) > 2):
            eval_score = 'f1'

        # create the list of options for the classification command
        options = [
            '-f %s' % (self.fm_dir),
            '-l %s' % (labeling_name),
            '-c %s' % (classifier),
            '-n %s' % (n_fold_cv),
            '-s %s' % (featsel),
            '-e %s' % (eval_score),
            '--classes %s' % (' '.join(class_ids)),
            '--features %s' % (' '.join(feat_ids.split(','))),
            '--standardize',
            '--timeout %i' % (self.TIMEOUT),
            '-o %s' % (out_dir)]

        # create command
        cmd = 'classification %s' % (' '.join(options))

        # create job file
        with open(os.path.join(self.job_waiting_dir, jobid), 'w') as fout:
            fout.write('%s\n' % (cmd))
            fout.write('%s\n' % (progress_f))
            fout.write('%s\n' % (error_f))

    def run_classify(self, cl_id, project_id):

        # obtain job id
        jobid = self.get_job_id()

        # read feature required for classifier
        settings_dict = self.get_classifier_settings(cl_id)
        feature_ids = settings_dict['feature_names']


        feature_cats = set()
        for f in feature_ids:
            fparts = f.split('_')
            if(len(fparts) < 3):
                feature_cats.add('_'.join(fparts[:1]))
            else:
                feature_cats.add('_'.join(fparts[:2]))

        # path to trained classifier file
        classifier_f = self.get_classifier_f

        # SWITCH TO OTHER PROJECT FOR FEATURE CALCULATION
        prev_proj = self.project_id
        self.set_project(project_id)

        # load feature extraction and obtain calculated feature categories
        fe = self.get_feature_extraction()
        calculated_feature_cats = fe.available_protein_featcat_ids()

        # determine missing feature categories
        missing_feature_cats = sorted(feature_cats - calculated_feature_cats)

        # queue feature calculation job if neccesary
        if(len(missing_feature_cats) > 0):
            self.run_feature_extraction(missing_feature_cats)

        # sleep for a second, to make sure feat calc job is first in queue
        time.sleep(2)

        # store path to feature matrix dir
        fm_dir = self.fm_dir        

        # SWITCH BACK TO ORIGINAL PROJECT
        self.set_project(prev_proj)

        # output dir
        out_d = os.path.join(self.get_cl_dir(cl_id), 'class_output')
        if not(os.path.exists(out_d)):
            os.mkdir(out_d)

        # output files
        progress_f = os.path.join(out_d, 'progress.txt')
        error_f = os.path.join(out_d, 'error.txt')
        
        # create the list of options for the classification command
        options = [
            '-f %s' % (fm_dir),
            '-c %s' % (self.get_cl_dir(cl_id))
        ]

        # create command
        cmd = 'classify %s' % (' '.join(options))

        # create job file
        with open(os.path.join(self.job_waiting_dir, jobid), 'w') as fout:
            fout.write('%s\n' % (cmd))
            fout.write('%s\n' % (progress_f))
            fout.write('%s\n' % (error_f))

