from util import sequtil


class Protein(object):

    def __init__(self, pid):

        self.pid = pid

        self.missense_mutations = []

        self.orf_sequence = None
        self.protein_sequence = None
        self.ss_sequence = None
        self.sa_sequence = None
        self.protein_structure = None

        self.rasa = None

        # rank score per residue
        self.msa_residue_rank = None
        self.msa_variability = None
        self.msa_coverage = None

        self.pfam_annotations = None

    def add_missense_mutation(self, mutation):
        self.missense_mutations.append(mutation)

    # set attributes, sequence data

    def set_orf_sequence(self, seq):
        self.orf_sequence = seq

    def set_protein_sequence(self, seq):
        self.protein_sequence = seq

    def set_protein_structure(self, struct):
        self.protein_structure = struct

    def set_ss_sequence(self, seq):
        self.ss_sequence = seq

    def set_sa_sequence(self, seq):
        serf.sa_sequence = seq

    def set_msa_data(self, msa_data):

        # 'unzip' lists
        if not(msa_data is None):
            i0, i1, r, cov, v0, v1, rank = zip(*msa_data)

            # check if sequence corresponds to protein sequence
            assert(''.join(r) == self.protein_sequence)
        else:
            # TODO check this!
            cov = [0.0] * len(self.protein_sequence)
            v1 = [[]] * len(self.protein_sequence)  # not sure about this...
            rank = [0.0] * len(self.protein_sequence)

        # store coverage, variability, and rank score
        self.msa_coverage = cov
        self.msa_variability = v1
        self.msa_residue_rank = rank

    def set_rasa(self, rasa):
        assert(rasa is None or type(rasa) == list)
        self.rasa = rasa

    def set_pfam_annotations(self, pfam_annotations):
        self.pfam_annotations = pfam_annotations

    # feature calculation functions

    def amino_acid_composition(self, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_composition(self.protein_sequence)
        else:
            return (list(sequtil.aa_unambiguous_alph),
                    sequtil.aa_unambiguous_name)

    def ss_composition(self, feature_ids=False):
        if not(feature_ids):
            return sequtil.ss_composition(self.ss_sequence)
        else:
            return (list(sequtil.ss_alph), sequtil.ss_name)

    def sa_composition(self, feature_ids=False):
        if not(feature_ids):
            return sequtil.sa_composition(self.sa_sequence)
        else:
            return (list(sequtil.sa_alph), sequtil.sa_name)
        

    def five_prime_amino_acid_count(self, seq_length, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_count(self.five_prime_seq(seq_length))
        else:
            return (list(sequtil.aa_unambiguous_alph),
                    sequtil.aa_unambiguous_name)

    def three_prime_amino_acid_count(self, seq_length, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_count(self.three_prime_seq(seq_length))
        else:
            return (list(sequtil.aa_unambiguous_alph),
                    sequtil.aa_unambiguous_name)

    def cluster_composition(self, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_cluster_composition(self.protein_sequence)
        else:
            return (sequtil.aa_subsets, sequtil.aa_subsets)

    def five_prime_cluster_count(self, seq_length, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_cluster_count(self.five_prime_seq(seq_length))
        else:
            return (list(sequtil.aa_unambiguous_alph),
                    sequtil.aa_subsets)

    def three_prime_cluster_count(self, seq_length, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_cluster_count(self.three_prime_seq(seq_length))
        else:
            return (list(sequtil.aa_unambiguous_alph),
                    sequtil.aa_subsets)

    # feature calculation help functions

    def five_prime_seq(self, seq_length):
        return self.protein_sequence[:seq_length]

    def three_prime_seq(self, seq_length):
        return self.protein_sequence[-seq_length:]

    def sequence_signal(self, scale, window, edge):
        return sequtil.seq_signal(self.protein_sequence, scale, window, edge)

    def pfam_family(self, position):
        return self.pfam_hmm_acc(position, 'Family')

    def pfam_domain(self, position):
        return self.pfam_hmm_acc(position, 'Domain')

    def pfam_repeat(self, position):
        return self.pfam_hmm_acc(position, 'Repeat')

    def pfam_hmm_acc(self, position, type):
        if not(self.pfam_annotations is None):
            for annotation in self.pfam_annotations:
                if(annotation.type_ == type):
                    if(position >= annotation.start_pos and
                            position <= annotation.end_pos):
                        # assuming no overlap, return the first one found
                        return annotation.hmm_acc
        return None

    def pfam_clan(self, position):
        if not(self.pfam_annotations is None):
            for annotation in self.pfam_annotations:
                if not(annotation.clan is None):
                    if(position >= annotation.start_pos and
                            position <= annotation.end_pos):
                        # assuming no overlap, return the first one found
                        return annotation.clan
        return None

    def pfam_active_residue(self, position):
        if not(self.pfam_annotations is None):
            for annotation in self.pfam_annotations:
                for r in annotation.active_residues:
                    if(r == position):
                        return True
        return False

    # check attribute availability functions (simple getters)

    def get_protein_sequence(self):
        return self.protein_sequence

    def get_orf_sequence(self):
        return self.orf_sequence

    def get_ss_sequence(self):
        return self.ss_sequence

    def get_sa_sequence(self):
        return self.sa_sequence

    def get_structure(self):
        return self.protein_structure

    def get_missense_mutations(self):
        return self.missense_mutations

    def get_rasa(self):
        return self.rasa


class Pfam(object):
    '''
    Class that contains Pfam annotation for a protein. Nothing more than a
    data store currently.
    '''

    def __init__(self, start_pos, end_pos, hmm_acc, hmm_name, type_,
            bit_score, e_value, clan, active_residues):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.hmm_acc = hmm_acc
        self.hmm_name = hmm_name
        self.type_ = type_
        self.bit_score = bit_score
        self.e_value = e_value
        self.clan = clan
        self.active_residues = active_residues

    def single_line_str(self):
        return '%i\t%i\t%s\t%s\t%s\t%.1f\t%e\t%s\t%s' % (self.start_pos,
                self.end_pos, self.hmm_acc, self.hmm_name, self.type_,
                self.bit_score, self.e_value, self.clan, self.active_residues)

    @classmethod
    def parse(self, s):
        tokens = s.split()
        start_pos = int(tokens[0])
        end_pos = int(tokens[1])
        hmm_acc = tokens[2]
        hmm_name = tokens[3]
        type_ = tokens[4]
        bit_score = float(tokens[5])
        e_value = float(tokens[6])
        clan = None if tokens[7] == 'None' else tokens[7]
        active_residues = eval(' '.join(tokens[8:]))

        return self(start_pos, end_pos, hmm_acc, hmm_name, type_, bit_score,
                e_value, clan, active_residues)
