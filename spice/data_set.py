import os

from spice.protein import Protein
from spice.mutation import MissenseMutation
from util import file_io
from util import sequtil


class ProteinDataSet(object):

    def __init__(self):

        # the list of protein objects
        self.proteins = []

        # the root directory, were the data will be stored
        self.root_dir = None

        # create sequence data source objects
        self.data_sources = DataSourceFactory().get_data_sources(self)
        self.ds_dict = dict([(ds.uid, ds) for ds in self.data_sources])

    def load_proteins(self, protein_ids_f):
        with open(protein_ids_f, 'r') as fin:
            protein_ids = [i for i in file_io.read_ids(fin)]
        self.set_proteins(protein_ids)

    def set_proteins(self, protein_ids):

        if not(len(protein_ids) == len(set(protein_ids))):
            raise ValueError('Duplicate ids encoutered.')

        assert(all([type(pid) == str for pid in protein_ids]))

        self.proteins = [Protein(pid) for pid in protein_ids]

    def set_root_dir(self, root_dir):
        self.root_dir = root_dir
        for ds in self.data_sources:
            ds.set_root_dir(root_dir)

    def get_proteins(self):
        return self.proteins

    def get_protein_ids(self):
        return [p.pid for p in self.proteins]

    def get_mutations(self):
        mutations = []
        for p in self.proteins:
            mutations.extend(p.missense_mutations)
        return mutations

    def get_mutation_ids(self):
        return [m.mid for m in self.get_mutations()]

    def read_data_source(self, src_id, data_path, mapping_file=None):
        assert(self.proteins)
        ds = self.ds_dict[src_id]
        ds.read_data(data_path, mapping_file=mapping_file,
                object_ids=self.get_protein_ids())
        self.propagate_data_source_data(ds)

    # TODO mapping? like in the function above
    def set_data_source(self, src_id, data, mapping_file=None):
        assert(self.proteins)
        ds = self.ds_dict[src_id]
        ds.set_data(data)
        self.propagate_data_source_data(ds)

    def propagate_data_source_data(self, data_source):
        '''
        Propagate the data that has been read/set by data source to the
        attributes of the Protein objects.
        '''
        if(data_source.data):
            for index, (data_id, data) in enumerate(data_source.data):
                assert(self.proteins[index].pid == data_id)
                data_source.set_data_func(self.proteins[index], data)

    def load_mutation_data(self, mutation_f):
        mut_data = [m for m in file_io.read_mutation(mutation_f)]
        self.set_mutation_data(mut_data)

    def set_mutation_data(self, mutation_data):
        '''
        The mutation data is a list of strings representing a mutation each.
        The strings are use to create MissenseMutation objects that are added
        to the Protein object it belongs to. Mutations in proteins that are
        not in the data set are neglected.
        '''
        assert(self.proteins)
        protein_dict = dict(zip(self.get_protein_ids(), self.proteins))

        '''
        for (pid, pos, fr, to, label, pep, pep_i, codons, codon_fr, codons_to,
             pdb_id, pdb_i) in mutation_data:
            if(pid in protein_dict.keys()):
                protein = protein_dict[pid]
                
                MissenseMutation(protein, pos, fr, to, label, pep, pep_i,
                                 codons, codon_fr, codons_to, pdb_id, pdb_i)
        '''
        for mismut_tuple in mutation_data:
            pid = mismut_tuple[0]
            if(pid in protein_dict.keys()):

                # fetch protein object
                protein = protein_dict[pid]

                # replace protein id by protein object in mutation tuple
                # TODO inefficient.... HACK
                mismut_list = list(mismut_tuple)
                mismut_list[0] = protein
                mismut_tuple = tuple(mismut_list)
                
                # create mutation object, which will imediately linked to the 
                # protein object
                MissenseMutation.from_tuple(mismut_tuple)

    def load(self):
        assert(self.root_dir)

        # initialize proteins from list of ids
        if(os.path.exists(self.protein_ids_f())):
            self.load_proteins(self.protein_ids_f())

        # load sequence data sources
        for ds in self.data_sources:
            ds.load()
            self.propagate_data_source_data(ds)

        # load mutation data
        if(os.path.exists(self.mutation_f())):
            self.load_mutation_data(self.mutation_f())

    def save(self):
        assert(self.root_dir)

        # create root dir, if not there yet
        if not(os.path.exists(self.root_dir)):
            os.makedirs(self.root_dir)

        # save protein ids, if available already
        if(self.proteins):

            # write protein ids to file
            file_io.write_ids(self.protein_ids_f(), self.get_protein_ids())

            # write mutation data to file, if any
            muts = [m.tuple_representation() for m in self.get_mutations()]
            if(muts):
                file_io.write_mutation(self.mutation_f(), muts)

        # save the data sources
        for ds in self.data_sources:
            ds.save()

    def protein_ids_f(self):
        return os.path.join(self.root_dir, 'protein_ids.txt')

    def mutation_f(self):
        return os.path.join(self.root_dir, 'missense.mut')


class DataSource():

    def __init__(self, data_set, uid, name, read_func, write_func,
            set_data_func, check_funcs, data_path, mapping_file):

        # callback data set
        self.data_set = data_set

        # data source id and name
        self.uid = uid
        self.name = name

        # function that can read and write the data to file
        self.read_func = read_func
        self.write_func = write_func
        self.check_funcs = check_funcs

        # link to the protein object attribute
        self.set_data_func = set_data_func

        # the data and mapping
        self.data = None
        self.data_mapping = None

        # path to data file or dir and mapping file for persistence
        self.data_path = data_path
        self.mapping_file = mapping_file

        # root dir is required for data persistence
        # TODO set to '' by default?
        self.root_dir = None

    def read_data(self, data_path, mapping_file=None, object_ids=None):
        '''
        Read data from a given location.

        data_path:
        Either a file or directory that contains the data.

        mapping_file:
        If the data source uses other ids or if the filenames
        do not correspond to the ids, a mapping is required to map to the
        uniprot ids we use.

        object_ds:
        If a list is provided, only data for the ids in the
        list will be read.
        '''

        if(object_ids is None):
            object_ids = self.data_set.get_protein_ids()

        # get mapping from our uniprot ids to data source ids
        if(mapping_file):
            object_to_data = [t for t in file_io.read_tuple_list(mapping_file,
                    (str, str))]
            # 'unzip' into list of mapped ids and list of data file names
            uni_othe_dict = dict(object_to_data)

        # read data, either from files in data_dir
        if(os.path.isdir(data_path)):

            # TODO mapping is required for this case... check that

            # get data ids or file names in the same order as the object ids
            data_fs = [uni_othe_dict[i] for i in object_ids]
            data_items = [a[1] for a in self.read_func(data_fs, data_path)]
            data = zip(object_ids, data_items)

            # set the data
            self.set_data(data, data_mapping=uni_othe_dict,
                    object_ids=object_ids)

        # or from a single data file
        else:
            data = [a for a in self.read_func(data_path)]

            # set the data
            if(mapping_file):
                data_dict = dict(data)
                data = [(i, data_dict[uni_othe_dict[i]]) for i in object_ids]
                self.set_data(data, data_mapping=uni_othe_dict,
                        object_ids=object_ids)
            else:
                self.set_data(data, object_ids=object_ids)

    def set_data(self, data, data_mapping=None, object_ids=None):

        self.data = data
        self.data_mapping = data_mapping

        # change from the other ids to our uniprot ids
        if(object_ids):
            d = dict(self.data)
            self.data = [(i, d[i]) for i in object_ids]

        for func in self.check_funcs:
            # check only the non-None items
            items_to_check = [s[1] for s in self.data if not s[1] is None]
            if (any(map(func, items_to_check))):
                self.data = None
                raise ValueError('Error in %s data, contains item that %s.' %
                    (self.name.lower(), ' '.join(func.__name__.split('_'))))

    def get_data_path(self):
        return(os.path.join(self.root_dir, self.data_path))

    def get_mapping_file(self):
        if(self.mapping_file):
            return(os.path.join(self.root_dir, self.mapping_file))
        else:
            return None

    def set_root_dir(self, root_dir):
        self.root_dir = root_dir

    def save(self):

        assert(self.root_dir)
        if(self.data):

            if(self.data_mapping):
                data_ids = [self.data_mapping[d[0]] for d in self.data]
                data_objects = [d[1] for d in self.data]
                out_data = zip(data_ids, data_objects)
            else:
                out_data = self.data

            # create root dir, if it is not their yet
            if not(os.path.exists(self.root_dir)):
                os.makedirs(self.root_dir)

            if(os.path.isdir(self.data_path)):

                # create data dir, if it is not there yet
                if not(os.path.exists(self.get_data_path())):
                    os.makedirs(self.get_data_path())

            if(self.data_mapping):

                # write mapping object id to data file mapping file
                #uni_ids = [d[0] for d in self.data]
                tuples = [(d[0], self.data_mapping[d[0]]) for d in self.data]
                file_io.write_tuple_list(self.get_mapping_file(), tuples)
            elif(self.get_mapping_file()):
                tuples = [(d[0], d[0]) for d in self.data]
                file_io.write_tuple_list(self.get_mapping_file(), tuples)

            # store the data
            self.write_func(self.get_data_path(), out_data)

    def load(self):

        dp = self.get_data_path()
        mf = self.get_mapping_file()

        if(os.path.exists(dp)):
            self.read_data(dp, mapping_file=mf)

    def available(self):
        return True if self.data else False

# TODO store this in configuration file
class DataSourceFactory(object):

    def __init__(self):

        self.data_source_ids = ['orf_seq', 'prot_seq', 'ss_seq', 'sa_seq',
                                #'prot_struct', 'residue_rasa', 'residue_rank',
                                'prot_struct', 'residue_rasa',
                                'msa', 'pfam', 'flex', 'interaction']

        # TODO somehow add function that checks if added data source is
        # consistent with allready available data, e.g. if the length of the
        # secondary structure sequences corresponds to the protein sequence
        # lengths.
        self.data_sources = {
            'prot_seq': ('Protein sequence',
                    file_io.read_fasta, file_io.write_fasta,
                    Protein.set_protein_sequence,
                    [
                        sequtil.is_empty,
                        sequtil.is_not_an_amino_acid_sequence
                    ], 'protein.fsa', None),
            'orf_seq': ('ORF sequence',
                    file_io.read_fasta, file_io.write_fasta,
                    Protein.set_orf_sequence,
                    [
                        sequtil.is_empty,
                        sequtil.is_not_a_nucleotide_sequence
                    ], 'orf.fsa', 'uni_orf.map'),
            'ss_seq': ('Secundary structure sequence',
                    file_io.read_fasta, file_io.write_fasta,
                    Protein.set_ss_sequence,
                    [
                        sequtil.is_empty,
                        sequtil.is_not_a_sec_struct_sequence
                    ], 'ss.fsa', 'uni_ss.map'),
            'sa_seq': ('Solvent accessible sequence',
                    file_io.read_fasta, file_io.write_fasta,
                    Protein.set_sa_sequence,
                    [
                        sequtil.is_empty,
                        sequtil.is_not_a_solv_access_sequence
                    ], 'sa.fsa', 'uni_sa.map'),
            'prot_struct': ('protein structure',
                    file_io.read_pdb_dir, file_io.write_pdb_dir,
                    Protein.set_protein_structure,
                    [
                    ], os.path.join('structure_data', 'pdb'), 'uni_pdb.map'),
            'residue_rasa': ('residue relative accessible surface area',
                    file_io.read_rasa_dir, file_io.write_rasa_dir,
                    Protein.set_rasa,
                    [
                    ], os.path.join('structure_data', 'rasa'),
                    'uni_rasa.map'),
            #'residue_rank': ('protein residue ranking',
            #        file_io.read_residue_rank_dir,
            #        file_io.write_residue_rank_dir,
            #        Protein.set_msa_data,
            #        [
            #        ], os.path.join('msa_data', 'residue_rank'),
            #        'uni_rank.map'),
            'msa': ('Multiple sequence alignment with homologous proteins',
                    file_io.read_msa_dir,
                    file_io.write_msa_dir,
                    Protein.set_msa,
                    [
                    ], os.path.join('msa_data', 'msa'),
                    'uni_msa.map'),
            'pfam': ('protein family data',
                    file_io.read_pfam, file_io.write_pfam,
                    Protein.set_pfam_annotations,
                    [], 'pfam.txt', None),
            'flex': ('backbone dynamics data',
                    file_io.read_flex, file_io.write_flex,
                    Protein.set_backbone_dynamics,
                    [], 'flex.txt', None),
            'interaction': ('interaction counts data',
                    file_io.read_interaction_counts,
                    file_io.write_interaction_counts,
                    Protein.set_interaction_counts,
                    [], 'interaction.txt', None)
        }

        # make sure that all ids are in the ids list
        assert(set(self.data_source_ids) ==
               set(self.data_sources.keys()))

    def get_data_sources(self, data_set):
        return [DataSource(data_set, sid, *self.data_sources[sid])
                for sid in self.data_source_ids]
