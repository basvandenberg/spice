#!/usr/bin/env python

"""
.. module:: featext

.. moduleauthor:: Bastiaan van den Berg <b.a.vandenberg@gmail.com>

"""

import os
import sys
import argparse
import traceback

import numpy

from spice import featmat
from spice import data_set
from spice import protein
from spice import mutation
from util import file_io


class FeatureExtraction(object):

    def __init__(self):

        # define file location if we want to store data
        self.root_dir = None

        # initialize empty feature matrices
        self.fm_protein = featmat.FeatureMatrix()
        self.fm_missense = featmat.FeatureMatrix()

        # initialize protein data set
        self.protein_data_set = data_set.ProteinDataSet()

        # initialize feature vectors
        self.fv_dict_protein = None
        self.fv_dict_missense = None

    def set_root_dir(self, root_dir):
        '''
        Set the root dir if you want to load or store the data to file.
        '''
        # store the root dir
        self.root_dir = root_dir

        # set the protein feature matrix root dir
        self.fm_protein_d = os.path.join(root_dir, 'feature_matrix_protein')
        #self.fm_protein.set_root_dir(self.fm_protein_d)

        # set the missense mutation feature matrix root dir
        self.fm_missense_d = os.path.join(root_dir, 'feature_matrix_missense')
        #self.fm_missense.set_root_dir(self.fm_missense_d)

        # set the protein data set root dir
        self.protein_data_set_d = os.path.join(root_dir, 'protein_data_set')
        self.protein_data_set.set_root_dir(self.protein_data_set_d)

    def set_protein_ids(self, protein_ids):
        # use protein ids to initiate protein objects in data set
        self.protein_data_set.set_proteins(protein_ids)
        # use the protein ids as object ids in the protein feature matrix
        self.fm_protein.object_ids = self.protein_data_set.get_protein_ids()

    def load_protein_ids(self, protein_ids_f):
        with open(protein_ids_f, 'r') as fin:
            protein_ids = [i for i in file_io.read_ids(fin)]
        self.set_protein_ids(protein_ids)

    def load_mutation_data(self, mutation_f):

        # add mutation data to protein data set
        self.protein_data_set.load_mutation_data(mutation_f)

        # and use the mutation ids as object ids in the protein feature matrix
        mut_ids = self.protein_data_set.get_mutation_ids()
        self.fm_missense.object_ids = mut_ids

    def calculate_protein_features(self, feat_vector_id):
        assert(self.fm_protein.object_ids)
        (fm, fids, fnames) = self.fv_dict_protein[feat_vector_id].calc_feats()
        self.fm_protein.add_features(fids, fm, feature_names=fnames)

    # TODO split in calculate and add function? or add remove from featmat
    def calculate_missense_features(self, feat_vector_id):
        assert(self.fm_protein.object_ids)
        (fm, fids, fnames) = self.fv_dict_missense[feat_vector_id].calc_feats()
        self.fm_missense.add_features(fids, fm, feature_names=fnames)

    def available_protein_feature_vectors(self):
        available = []
        if(self.fm_protein.feature_ids):
            feat_ids_set = set(self.fm_protein.feature_ids)
            for featvec in self.fv_dict_protein.values():
                if(set(featvec.feat_ids) <= feat_ids_set):
                    available.append(featvec)
        return available

    def protein_feat_id_to_name_dict(self):
        result = {}
        for featvec in self.fv_dict_protein.values():
            featvec_dict = featvec.feat_name_dict()
            for key in featvec_dict.keys():
                result[key] = (featvec.name, featvec_dict[key])
        return result

    def get_protein_feature_vector_ids(self):
        return ProteinFeatureVectorFactory.FEATVEC_IDS

    def load(self):

        assert(self.root_dir)

        # load the feature matrices
        fmp = featmat.FeatureMatrix.load_from_dir(self.fm_protein_d)
        self.fm_protein = fmp

        fmm = featmat.FeatureMatrix.load_from_dir(self.fm_missense_d)
        self.fm_missense = fmm

        # load protein data set
        self.protein_data_set.load()

        # create protein feature vectors object
        if(self.protein_data_set.get_proteins()):
            self.fv_dict_protein = ProteinFeatureVectorFactory().\
                    get_feature_vectors(self.protein_data_set.get_proteins())

        # create mutation feature vector object
        if(self.protein_data_set.get_mutations()):
            self.fv_dict_missense = MutationFeatureVectorFactory().\
                    get_feature_vectors(self.protein_data_set.get_mutations())

    def save(self):

        assert(self.root_dir)

        # create root dir, if not available
        if not(os.path.exists(self.root_dir)):
            os.makedirs(self.root_dir)

        # save feature matrix
        self.fm_protein.save_to_dir(self.fm_protein_d)
        self.fm_missense.save_to_dir(self.fm_missense_d)

        # save protein data set
        self.protein_data_set.save()

    def __str__(self):
        return '%s\n%s\n' % (str(self.fm_protein), str(self.fm_missense))

    def plot_mutation_environment_signals(self, env_window, scale, sig_window,
            edge, labeling_name):

        if(sig_window > env_window):
            raise ValueError('Signal window (sig_window) must be equal or ' +
                    'smaller than environment window (env_window).')

        mutations = self.protein_data_set.get_mutations()
        labeling = self.fm_missense.labeling_dict[labeling_name]

        # TODO use any labeling... now hacked to my mut labels
        indices_0 = labeling.object_indices_per_class['neutral']
        indices_1 = labeling.object_indices_per_class['disease']

        nrows = env_window - (sig_window - 1)
        ncols = len(mutations)

        data = numpy.zeros((nrows, ncols))

        for index, mut in enumerate(mutations):
            data[:, index] = mut.environment_signal(env_window, scale,
                    sig_window, edge)

        # split data per label # TODO implement multiple labels
        data0 = data[:, indices_0]
        data1 = data[:, indices_1]

        from matplotlib import pyplot
        pyplot.plot(data0[:, range(300)], color="#55ff55", alpha=0.1)
        pyplot.plot(data1[:, range(300)], color="#ff5555", alpha=0.1)
        #pyplot.plot(data0, color="#55ff55")
        #pyplot.plot(data1, color="#ff5555")
        pyplot.show()


class FeatureVector():

    def __init__(self, uid, object_list, name, feature_func, kwargs,
            required_data=None):

        assert(object_list)

        self.uid = uid
        self.name = name

        self.object_list = object_list
        self.feature_func = feature_func
        self.kwargs = kwargs

        self.required_data = required_data

        (ids, names) = self.feature_func(self.object_list[0],
                feature_ids=True, **kwargs)

        self.short_ids = ids
        self.short_names = names

        self.feat_ids = ['%s_%s' % (self.uid, i) for i in ids]
        #self.feat_names = ['%s %s' % (self.name, n) for n in names]
        self.feat_names = names

    def required_data_available(self, get_data_func, all_objects=True):
        '''
        get_data_func: is a simple getter that should be available for the
        objects in self.object_list
        all_objects: if False, then any is assumed.
        '''
        if(all_objects):
            return(all([get_data_func(obj) for obj in self.object_list]))
        else:
            return(any([get_data_func(obj) for obj in self.object_list]))

    def data_availability(self):
        available_data = []
        missing_data = []
        for rdata, _all in self.required_data:
            data_name = ' '.join(rdata.__name__.split('_')[1:])
            if(self.required_data_available(rdata, _all)):
                available_data.append(data_name)
            else:
                missing_data.append(data_name)
        return (available_data, missing_data)

    def calc_feats(self):

        # init empty feature matrix
        fm = numpy.empty((len(self.object_list), len(self.feat_ids)))

        # fill the matrix
        for index, o in enumerate(self.object_list):
            fm[index, :] = self.feature_func(o, **self.kwargs)
        return(fm, self.feat_ids, self.feat_names)

    def feat_name_dict(self):
        return dict(zip(self.feat_ids, self.feat_names))


class FeatureVectorFactory(object):

    def get_feature_vectors(self, object_list):
        return dict(zip(self.FEATVEC_IDS,
                [FeatureVector(fid, object_list, *self.feature_vectors[fid])
                for fid in self.FEATVEC_IDS]))


class MutationFeatureVectorFactory(FeatureVectorFactory):

    FEATVEC_IDS = [
        'mutvec', 'mutggsigdiff', 'mutggbsigdiff',
        'seqenv19',
        'msa', 'msaggsigdiff',
        'bbang', 'rasa',
        'pfam', 'flex', 'codonvec', 'codonenv19'
    ]

    def __init__(self):

        self.feature_vectors = {
            'mutvec': ('mutation vector',
                mutation.MissenseMutation.mutation_vector, {}),
            'mutggsigdiff': ('mutation georgiev signal difference',
                mutation.MissenseMutation.georgiev_signal_diff, {}),
            'mutggbsigdiff': ('mutation georgiev blosum signal difference',
                mutation.MissenseMutation.georgiev_blosum_signal_diff, {}),
            'seqenv19': ('sequence environment amino acid counts',
                mutation.MissenseMutation.seq_env_aa_count, {'window': 19}),
            #'msa': ('msa-based',
            #    mutation.MissenseMutation.msa_based, {}),
            'msa': ('msa-based',
                mutation.MissenseMutation.msa, {}),
            'msaggsigdiff': ('msa georgiev signal difference',
                mutation.MissenseMutation.msa_scale_diff, {}),
            'bbang': ('backbone angles',
                mutation.MissenseMutation.backbone_angles, {}),
            'rasa': ('relative accessible surface area',
                mutation.MissenseMutation.solv_access, {}),
            'pfam': ('pfam annotation',
                mutation.MissenseMutation.pfam_annotation, {}),
            'flex': ('backbone_dynamics',
                mutation.MissenseMutation.residue_flexibility, {}),
            'codonvec': ('from codon vector',
                mutation.MissenseMutation.from_codon_vector, {}),
            'codonenv19': ('sequence environment codon counts',
                mutation.MissenseMutation.seq_env_codon_count, {})
        }

        # make sure that all ids are in the ids list
        assert(set(self.FEATVEC_IDS) == set(self.feature_vectors.keys()))


class ProteinFeatureVectorFactory(FeatureVectorFactory):

    FEATVEC_IDS = [
        'aac', 'clc',
        'ssc', 'ssaac',
        'sac', 'saaac',
        'codc', 'codu',
        '5p75aac', '3p75aac', '5p75clc', '3p75clc',
        'sigavg', 'sigpeak',
        'len'
    ]

    def __init__(self):

        self.feature_vectors = {
            'aac': ('amino acid composition',
                protein.Protein.amino_acid_composition, {},
                [(protein.Protein.get_protein_sequence, True)]),
            'clc': ('cluster composition',
                protein.Protein.cluster_composition, {},
                [(protein.Protein.get_protein_sequence, True)]),
            'ssc': ('secondary structure composition',
                protein.Protein.ss_composition, {},
                [(protein.Protein.get_ss_sequence, True)]),
            'ssaac': ('secondary structure amino acid composition',
                protein.Protein.ss_aa_composition, {},
                [(protein.Protein.get_ss_sequence, True),
                (protein.Protein.get_protein_sequence, True)]),
            'sac': ('solvent accessibility composition',
                protein.Protein.sa_composition, {},
                [(protein.Protein.get_sa_sequence, True)]),
            'saaac': ('solvent accessibility amino acid composition',
                protein.Protein.sa_aa_composition, {},
                [(protein.Protein.get_sa_sequence, True),
                (protein.Protein.get_protein_sequence, True)]),
            'codc': ('codon composition',
                protein.Protein.codon_composition, {},
                [(protein.Protein.get_orf_sequence, True)]),
            'codu': ('codon usage',
                protein.Protein.codon_usage, {},
                [(protein.Protein.get_orf_sequence, True)]),
            '5p75aac': ('5-prime 75 AA count',
                protein.Protein.five_prime_amino_acid_count,
                {'seq_length': 75},
                [(protein.Protein.get_protein_sequence, True)]),
            '3p75aac': ('3-prime 75 AA count',
                protein.Protein.three_prime_amino_acid_count,
                {'seq_length': 75},
                [(protein.Protein.get_protein_sequence, True)]),
            '5p75clc': ('5-prime 75 AA cluster count',
                protein.Protein.five_prime_cluster_count,
                {'seq_length': 75},
                [(protein.Protein.get_protein_sequence, True)]),
            '3p75clc': ('3-prime 75 AA cluster count',
                protein.Protein.three_prime_cluster_count,
                {'seq_length': 75},
                [(protein.Protein.get_protein_sequence, True)]),
            'sigavg': ('average signal value',
                protein.Protein.average_signal,
                {'window': 5, 'edge':0.5},
                [(protein.Protein.get_protein_sequence, True)]),
            'sigpeak': ('signal value peaks area', 
                protein.Protein.signal_peaks_area,
                {'window': 5, 'edge':0.5, 'threshold':1.0},
                [(protein.Protein.get_protein_sequence, True)]),
            'len': ('protein length',
                protein.Protein.length, {},
                [(protein.Protein.get_protein_sequence, True)])
        }

        # make sure that all ids are in the ids list
        assert(set(self.FEATVEC_IDS) == set(self.feature_vectors.keys()))


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()

    # required to provide a root directory for the application
    parser.add_argument('-r', '--root', required=True)

    # initialize a new project
    parser.add_argument('-i', '--init', action='store_true', default=False)

    # set protein ids
    parser.add_argument('-u', '--uniprot_ids')

    # add path to mutation data
    parser.add_argument('-m', '--missense_mutations')

    # optional, only needed for classification stuff (hists, ttest, ...)
    parser.add_argument('-l', '--labels', nargs=3, action='append')

    # add path to protein sequence data sources (that use the uniprot ids)
    parser.add_argument('--protein_sequence_data')

    # add path to pfam annotation data
    parser.add_argument('--pfam_data')

    # add backbone dynamics data
    parser.add_argument('--flex_data')

    # data sources that have a mapping from their ids to uniprot ids
    parser.add_argument('--orf_sequence_data', nargs=2)
    parser.add_argument('--structure_data', nargs=2)
    parser.add_argument('--rasa_data', nargs=2)
    #parser.add_argument('--residue_rank_data', nargs=2)
    parser.add_argument('--msa_data', nargs=2)

    # features to be calculated
    parser.add_argument('--missense_features', nargs='+', default=None)
    parser.add_argument('--protein_features', nargs='+', default=None)

    # TODO implement this
    # user should provide 2 paths, one to feature matrix, one to feature ids
    parser.add_argument('--custom_missense_features', nargs=2)
    parser.add_argument('--custom_protein_features', nargs=2)

    # delete feature matrix TODO do this without loading the data set???
    parser.add_argument('--delete_feature_matrices', action='store_true',
            default=False)

    args = parser.parse_args()

    # create feature extraction object
    fe = FeatureExtraction()
    fe.set_root_dir(args.root)

    # initialize new project
    if(args.init):

        if(os.path.exists(args.root) and os.listdir(args.root)):
            print('\nUnable to initialize a new project in '
                  '%s, the directory is not empty.\n' % (args.root))
            sys.exit()
        else:
            fe.save()
            print('\nNew project created in %s\n' % (args.root))

    # try to load project
    if not(os.path.exists(args.root)):
        print '\nDirectory %s does not exist' % (args.root)
        print 'Use --init if you want to create a new project.\n'
        sys.exit()
    else:
        try:
            print('\nLoading data...')
            fe.load()
            print('Done.')
        except Exception, e:
            print '\nError while loading project: %s\n' % (e)
            raise

    # initialize proteins using a list of ids
    if(args.uniprot_ids):

        if(fe.protein_data_set.proteins):
            print('\nProteins are allready set.\n')
            sys.exit()
        else:
            try:
                fe.load_protein_ids(args.uniprot_ids)
            except IOError as e:
                print '\nNo such file: %s\n' % (e)
                sys.exit()
            except Exception, e:
                print '\nError in object ids file: %s\n' % (e)
                sys.exit()

        fe.save()

    # set labels
    if(args.labels):

        label_types = ['protein', 'missense']

        for (label_type, label_name, label_path) in args.labels:

            if not(label_type in label_types):
                print '\nWrong label type: %s' % (label_type)
                print 'Must be one of: %s\n' % (', '.join(label_types))
                sys.exit(1)
            try:
                if(label_type == 'protein'):
                    fe.fm_protein.add_labeling_from_file(label_name,
                                                         label_path)
                elif(label_type == 'missense'):
                    fe.fm_missense.add_labeling_from_file(label_name,
                                                          label_path)
                else:
                    print '\nWrong label type, error should not occur...\n'
                    sys.exit(1)
            except IOError, e:
                print traceback.format_exc()
                sys.exit(1)
            except ValueError, e:
                print traceback.format_exc()
                sys.exit(1)
            except Exception, e:
                print traceback.format_exc()
                sys.exit(1)

        fe.save()

    # add protein sequence data (obtain from fasta file using uniprot ids)
    if(args.protein_sequence_data):

        ds_name = 'prot_seq'
        prot_ds = fe.protein_data_set
        ds_path = args.protein_sequence_data

        # the try is not realy neccasary anymore now... TODO remove
        try:
            ds = prot_ds.ds_dict[ds_name]
        except KeyError:
            print('\nNo such data source: %s\n' % (ds_name))
            sys.exit()

        if(ds.available()):
            print('\nData source already available: %s\n' % (ds_name))
            sys.exit()
        else:
            try:
                prot_ds.read_data_source(ds_name, ds_path, None)
            except IOError as e:
                print '\nData source io error: %s\n\n%s' % (ds_name, e)
                sys.exit()
            except ValueError as e:
                print '\nData source value error: %s\n\n%s' % (ds_name, e)
                sys.exit()
            except Exception as e:
                print '\nData source exception: %s\n\n%s' % (ds_name, e)
                print sys.exc_info()[0]
                sys.exit()

        fe.save()

    # add pfam annotation data
    if(args.pfam_data):

        ds_name = 'pfam'
        prot_ds = fe.protein_data_set
        ds_path = args.pfam_data

        try:
            ds = prot_ds.ds_dict[ds_name]
        except KeyError:
            print('\nNo such data source: %s\n' % (ds_name))
            sys.exit()

        if(ds.available()):
            print('\nData source already available: %s\n' % (ds_name))
            sys.exit()
        else:
            try:
                prot_ds.read_data_source(ds_name, ds_path, None)
            except IOError as e:
                print '\nData source io error: %s\n\n%s' % (ds_name, e)
                sys.exit()
            except ValueError as e:
                print '\nData source value error: %s\n\n%s' % (ds_name, e)
                sys.exit()
            except Exception as e:
                print '\nData source exception: %s\n\n%s' % (ds_name, e)
                print sys.exc_info()[0]
                sys.exit()

        fe.save()

    # add backbone dynamics data
    if(args.flex_data):

        ds_name = 'flex'
        prot_ds = fe.protein_data_set
        ds_path = args.flex_data

        try:
            ds = prot_ds.ds_dict[ds_name]
        except KeyError:
            print('\nNo such data source: %s\n' % (ds_name))
            sys.exit(1)

        if(ds.available()):
            print('\nData source already available: %s\n' % (ds_name))
            sys.exit(1)
        else:
            try:
                prot_ds.read_data_source(ds_name, ds_path, None)
            except IOError as e:
                print traceback.format_exc()
                sys.exit(1)
            except ValueError as e:
                print traceback.format_exc()
                sys.exit(1)
            except Exception as e:
                print traceback.format_exc()
                sys.exit(1)

        fe.save()

    # add orf sequence data,
    if(args.orf_sequence_data):

        (uni_orf_map_f, ds_path) = args.orf_sequence_data

        ds_name = 'orf_seq'
        prot_ds = fe.protein_data_set

        #ds_path = args.protein_sequence_data
        #ds_name = 'protseq'

        # the try is not realy neccasary anymore now... TODO remove
        try:
            ds = prot_ds.ds_dict[ds_name]
        except KeyError:
            print('\nNo such data source: %s\n' % (ds_name))
            sys.exit()

        if(ds.available()):
            print('\nData source already available: %s\n' % (ds_name))
            sys.exit()
        else:
            try:
                prot_ds.read_data_source(ds_name, ds_path,
                        mapping_file=uni_orf_map_f)
            except IOError as e:
                print '\nData source io error: %s\n\n%s' % (ds_name, e)
                print traceback.format_exc()
                sys.exit(1)
            except ValueError as e:
                print '\nData source value error: %s\n\n%s' % (ds_name, e)
                print traceback.format_exc()
                sys.exit(1)
            except Exception as e:
                print '\nData source exception: %s\n\n%s' % (ds_name, e)
                print traceback.format_exc()
                sys.exit(1)

        fe.save()

    # add structure data
    if(args.structure_data):

        (ids_file, pdb_dir) = args.structure_data

        prot_ds = fe.protein_data_set
        ds = prot_ds.ds_dict['prot_struct']

        if(ds.available()):
            print('\nProtein structure data already available.\n')
            sys.exit()
        else:
            try:
                prot_ds.read_data_source('prot_struct', pdb_dir, ids_file)
            except IOError as e:
                print '\nData source io error: prot_struct\n'
                print traceback.format_exc()
                sys.exit()
            except ValueError as e:
                print '\nData source value error: prot_struct\n'
                print traceback.format_exc()
                sys.exit()
            except Exception as e:
                print '\nData source exception: prot_struct\n'
                print traceback.format_exc()
                sys.exit()

        fe.save()

    # add solvent accessibilty data
    if(args.rasa_data):

        (ids_file, rasa_dir) = args.rasa_data

        prot_ds = fe.protein_data_set
        ds = prot_ds.ds_dict['residue_rasa']

        if(ds.available()):
            print('\nSolvent accessibility data already available.\n')
            sys.exit()
        else:
            try:
                prot_ds.read_data_source('residue_rasa', rasa_dir, ids_file)
            except IOError as e:
                print '\nData source io error: residue_rasa\n'
                print traceback.format_exc()
                sys.exit()
            except ValueError as e:
                print '\nData source value error: residue_rasa\n'
                print traceback.format_exc()
                sys.exit()
            except Exception as e:
                print '\nData source exception: residue_rasa\n'
                print traceback.format_exc()
                sys.exit()

        fe.save()

    # add residue rank data
    '''
    if(args.residue_rank_data):

        (ids_file, rank_dir) = args.residue_rank_data

        prot_ds = fe.protein_data_set
        ds = prot_ds.ds_dict['residue_rank']

        if(ds.available()):
            print('\nProtein residue rank data already available.\n')
            sys.exit()
        else:
            try:
                prot_ds.read_data_source('residue_rank', rank_dir, ids_file)
            except IOError as e:
                print traceback.format_exc()
                sys.exit(1)
            except ValueError as e:
                print traceback.format_exc()
                sys.exit(1)
            except Exception as e:
                print traceback.format_exc()
                sys.exit(1)

        fe.save()
    '''

    # add MSA data
    if(args.msa_data):

        (ids_file, msa_dir) = args.msa_data

        prot_ds = fe.protein_data_set
        ds = prot_ds.ds_dict['msa']

        if(ds.available()):
            print('\nProtein msa data already available.\n')
            sys.exit()
        else:
            try:
                prot_ds.read_data_source('msa', msa_dir, ids_file)
            except IOError as e:
                print traceback.format_exc()
                sys.exit(1)
            except ValueError as e:
                print traceback.format_exc()
                sys.exit(1)
            except Exception as e:
                print traceback.format_exc()
                sys.exit(1)

        fe.save()

    # add missense mutation data
    if(args.missense_mutations):

        # check if protein sequences are available
        if not(fe.protein_data_set.ds_dict['prot_seq'].available()):
            print('\nMutation data can only be added if protein sequences' +
                    ' are available.\n')
            sys.exit()

        else:

            # check if mutations are not allready present
            if(fe.protein_data_set.get_mutations()):
                print('\nMutation data already available.\n')
                sys.exit()
            else:
                try:
                    fe.load_mutation_data(args.missense_mutations)
                except IOError as e:
                    print traceback.print_exc()
                    sys.exit(1)
                except ValueError as e:
                    print traceback.print_exc()
                    sys.exit(1)
                except Exception as e:
                    print traceback.print_exc()
                    sys.exit(1)
        fe.save()

    # calculate features
    if(args.missense_features):

        for feature_vector in args.missense_features:
            try:
                fe.calculate_missense_features(feature_vector)
            except ValueError, e:
                #print('\nFeature category error: %s\n' % (e))
                print traceback.print_exc()
                #sys.exit()
                raise e
            except Exception as e:
                print('\nFeature calculation error: %s\n' % (e))
                print traceback.print_exc()
                sys.exit(1)

        fe.save()

    if(args.protein_features):

        for feature_vector in args.protein_features:

            try:
                fe.calculate_protein_features(feature_vector)
            except ValueError, e:
                print('\nFeature category error: %s\n' % (e))
                print traceback.print_exc()
                raise e
            except Exception as e:
                print('\nFeature calculation error: %s\n' % (e))
                print traceback.print_exc()
                raise e

        fe.save()

    # add custom features
    if(args.custom_missense_features):
        feat_ids_f, feat_mat_f = args.custom_missense_features
        # TODO implement

    if(args.custom_protein_features):
        feat_ids_f, feat_mat_f = args.custom_protein_features
        # TODO implement

    if(args.delete_feature_matrices):
        fe.delete_feature_matrices()
        fe.save()
