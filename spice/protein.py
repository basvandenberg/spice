import math
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

        # TODO depricate
        # rank score per residue
        self.msa_residue_rank = None
        #self.msa_variability = None
        self.msa_coverage = None

        # updated own hhblits MSA, list of aligned sequences, first is this seq
        self.msa = None

        self.pfam_annotations = None
        self.backbone_dynamics = None

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
        self.sa_sequence = seq

    # TODO depricate
    def set_msa_data(self, msa_data):

        # 'unzip' lists
        if not(msa_data is None):
            i0, i1, r, cov, v0, v1, rank = zip(*msa_data)

            # check if sequence corresponds to protein sequence
            #assert(''.join(r) == self.protein_sequence)
            # I currently allow for at most 5 residue differences...
            assert(len(r) == len(self.protein_sequence))
        else:
            # TODO default values!!! check this!
            cov = [0.0] * len(self.protein_sequence)
            v1 = [[]] * len(self.protein_sequence)  # not sure about this...
            rank = [0.0] * len(self.protein_sequence)

        # store coverage, variability, and rank score
        self.msa_coverage = cov
        self.msa_variability = v1
        self.msa_residue_rank = rank

    def set_msa(self, msa):
        '''
        msa is list with (equal length) aligned sequences. The first sequence
        is the sequence of the protein (without gaps).
        '''

        if(msa is None):
            # if not available, use own sequence as only sequence
            msa = [self.protein_sequence]

        else:
            # checks
            if not(msa[0] == self.protein_sequence):
                raise ValueError('First sequence in MSA does not correspond ' +
                                 'to this protein sequence')
            if not(all([len(msa[0]) == len(m) for m in msa])):
                raise ValueError('Not all sequences in MSA have the same ' +
                                 'length')

        # store data
        self.msa = msa

    def set_rasa(self, rasa):
        assert(rasa is None or type(rasa) == list)
        self.rasa = rasa

    def set_pfam_annotations(self, pfam_annotations):
        self.pfam_annotations = pfam_annotations

    def set_backbone_dynamics(self, backbone_dynamics):
        assert(type(backbone_dynamics) == list)
        assert(len(backbone_dynamics) == len(self.protein_sequence))
        self.backbone_dynamics = backbone_dynamics

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

    def ss_aa_composition(self, feature_ids=False):
        if not(feature_ids):
            return sequtil.ss_aa_composition(self.protein_sequence,
                                             self.ss_sequence)
        else:
            ids = ['%s%s' % (s, a) for s in sequtil.ss_alph
                   for a in sequtil.aa_unambiguous_alph]
            names = ['%s-%s' % (s, a) for s in sequtil.ss_name
                     for a in sequtil.aa_unambiguous_name]
            return (ids, names)

    def sa_aa_composition(self, feature_ids=False):
        if not(feature_ids):
            return sequtil.sa_aa_composition(self.protein_sequence,
                                             self.sa_sequence)
        else:
            ids = ['%s%s' % (s, a) for s in sequtil.sa_alph
                   for a in sequtil.aa_unambiguous_alph]
            names = ['%s-%s' % (s, a) for s in sequtil.sa_name
                     for a in sequtil.aa_unambiguous_name]
            return (ids, names)

    def five_prime_amino_acid_count(self, seq_length=75, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_count(self.five_prime_seq(seq_length))
        else:
            return (list(sequtil.aa_unambiguous_alph),
                    sequtil.aa_unambiguous_name)

    def three_prime_amino_acid_count(self, seq_length=75, feature_ids=False):
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

    def five_prime_cluster_count(self, seq_length=75, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_cluster_count(self.five_prime_seq(seq_length))
        else:
            return (list(sequtil.aa_unambiguous_alph),
                    sequtil.aa_subsets)

    def three_prime_cluster_count(self, seq_length=75, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_cluster_count(self.three_prime_seq(seq_length))
        else:
            return (list(sequtil.aa_unambiguous_alph),
                    sequtil.aa_subsets)

    def codon_composition(self, feature_ids=False):
        if not(feature_ids):
            return sequtil.codon_composition(self.orf_sequence)
        else:
            names = ['%s (%s)' % (c, sequtil.translate(c))
                     for c in sequtil.codons_unambiguous]
            return (sequtil.codons_unambiguous, names)

    def codon_usage(self, feature_ids=False):
        if not(feature_ids):
            return sequtil.codon_usage(self.orf_sequence)
        else:
            names = ['%s (%s)' % (c, sequtil.translate(c))
                     for c in sequtil.codons_unambiguous]
            return (sequtil.codons_unambiguous, names)

    def average_signal(self, window=9, edge=0, feature_ids=False):

        if not(feature_ids):

            scales = sequtil.georgiev_scales
            result = []

            for scale in scales:
                result.append(sequtil.avg_seq_signal(
                              self.protein_sequence, scale, window, edge))

            return result
        else:
            return (['%02d' % (i) for i in range(19)],
                    ['Georgiev %i' % (i) for i in range(19)])

    def signal_peaks_area(self, window=9, edge=0, threshold=1.0,
                          feature_ids=False):

        if not(feature_ids):

            scales = sequtil.georgiev_scales
            result = []

            for scale in scales:
                top, bot = sequtil.auc_seq_signal(self.protein_sequence, scale,
                                                  window, edge, threshold)
                result.append(top)
                result.append(bot)

            return result
        else:
            return (['%02d%s' % (i, s) for s in ['t', 'b'] for i in range(19)],
                    ['Georgiev %i %s' % (i, s) for s in ['top', 'bottom']
                    for i in range(19)])

    def length(self, feature_ids=False):
        if not(feature_ids):
            return [len(self.protein_sequence)]
        else:
            return (['len'], ['Protein length'])

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

    def msa_column(self, position, with_gaps=True):
        '''
        Returns the aligned amino acids of the give positions, without the
        gaps (-).
        '''
        index = position - 1

        if(with_gaps):
            return [s[index] for s in self.msa]
        else:
            return [s[index] for s in self.msa if not s[index] == '-']

    def msa_variability(self, position, with_gaps=False):
        '''
        Returns the set of (unambiguous) amino acid letters found on the given
        position in the multiple sequence alignment. All other letters (exept
        the gap character '-') are disregarded.

        with_gaps: If set to True, a gap is also part of the column variability
        '''

        # amino acids + gap
        aas = sequtil.aa_unambiguous_alph + '-'
        column_letters_set = set(self.msa_column(position, with_gaps))
        return [l for l in column_letters_set if l in aas]

    def msa_fraction(self, position, letter, with_gaps):
        '''
        TODO: what to do if no aligned seqs, or only few...
        !!! with or without gaps...
        '''
        # obtain all letters on this position in the MSA
        col = self.msa_column(position, with_gaps=with_gaps)

        if(len(col) <= 1):
            assert(len(col) == 1)
            return 0.5
        else:
            # return the fraction of letter
            return float(col.count(letter)) / len(col)

    def msa_conservation_index(self, position):
        '''
        SNPs&GO conservation index, I don't understand the formula...
        '''
        pass

    def msa_entropy21(self, position, with_gaps):

        # amino acids + gap
        aas = sequtil.aa_unambiguous_alph + '-'

        # obtain MSA column letters
        col = self.msa_column(position, with_gaps=with_gaps)

        # is not always the case...
        #assert(all([c in aas for c in col]))

        if(len(col) <= 1):

            assert(len(col) == 1)

            # default entropy in case of no aligned sequences
            entropy = 0.5

            # TODO num seqs < some threshold? Do some other default thing?
        else:

            n = len(col)
            k = len(aas)  # should be 21, 20 amino acids + 1 gap

            # fraction per letter
            na_list = [col.count(l) for l in set(col)]
            pa_list = [float(na) / n for na in na_list]
            na_log_sum = sum([pa * math.log(pa, 2) for pa in pa_list])

            # calculate entropy and return that
            entropy = (-1.0 * na_log_sum) / math.log(min(n, k), 2)

        return entropy

    # check attribute availability functions (simple getters)

    def get_protein_sequence(self):
        return self.protein_sequence

    def get_orf_sequence(self):
        return self.orf_sequence

    def get_ss_sequence(self):
        return self.ss_sequence

    def get_sa_sequence(self):
        return self.sa_sequence

    def get_msa(self):
        return self.msa

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
        return '%i\t%i\t%s\t%s\t%s\t%.1f\t%e\t%s\t%s' % (
            self.start_pos, self.end_pos, self.hmm_acc, self.hmm_name,
            self.type_, self.bit_score, self.e_value, self.clan,
            self.active_residues)

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
