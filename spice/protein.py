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

        # protein interaction counts (maybe combine in tuple, index as consts)
        self.ppi_count = None
        self.metabolic_count = None
        self.genetic_count = None
        self.phosphorylation_count = None
        self.regulatory_count = None
        self.signaling_count = None

    def add_missense_mutation(self, mutation):
        self.missense_mutations.append(mutation)

    # set attributes, sequence data
    # TODO turn this into proper setters...

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
        self.pfam_annotations = [Pfam(a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                                 a[7], a[8]) for a in pfam_annotations]

    def set_backbone_dynamics(self, backbone_dynamics):
        assert(type(backbone_dynamics) == list)
        assert(len(backbone_dynamics) == len(self.protein_sequence))
        self.backbone_dynamics = backbone_dynamics

    def set_interaction_counts(self, interaction_counts):
        self.ppi_count = interaction_counts[0]
        self.metabolic_count = interaction_counts[1]
        self.genetic_count = interaction_counts[2]
        self.phosphorylation_count = interaction_counts[3]
        self.regulatory_count = interaction_counts[4]
        self.signaling_count = interaction_counts[5]

    ###########################################################################
    # feature calculation functions
    ###########################################################################

    def amino_acid_composition(self, num_segments, feature_ids=False):

        if not(feature_ids):

            return sequtil.aa_composition(self.protein_sequence, num_segments)
        else:

            feat_ids = []
            feat_names = []

            for si in xrange(1, num_segments + 1):
                for aa in sequtil.aa_unambiguous_alph:
                    feat_ids.append('%s%i' % (aa, si))
                    feat_names.append(
                        'amino acid %s, segment %i' % (aa, si))

            return (feat_ids, feat_names)

    def prime_amino_acid_count(self, prime, length, feature_ids=False):

        if not(feature_ids):

            return sequtil.aa_count(self.prime_seq(prime, length))
        else:
            aa_alph = sequtil.aa_unambiguous_alph
            feat_ids = ['%ip%s' % (prime, aa) for aa in aa_alph]
            feat_names = ["%i' amino acid count %s" % (prime, aa)
                          for aa in aa_alph]
            return (feat_ids, feat_names)

    def _parse_scales(self, scales):

        if(type(scales) == str):
            try:
                scales = int(scales)
            except ValueError:
                pass

        # 'parse' the scale parameter
        if(scales == 'gg'):
            # retrieve set of georgiev scales
            scale_list = sequtil.get_georgiev_scales()
            scale_ids = ['gg%i' % (i) for i in xrange(1, len(scale_list) + 1)]
            scale_names = ['Georgiev scale %i' % (i)
                           for i in xrange(1, len(scale_list) + 1)]
        elif(type(scales) == int):
            scale_list = [sequtil.get_aaindex_scale(scales)]
            scale_ids = ['aai%i' % (scales)]
            scale_names = ['amino acid index %i' % (scales)]
        elif(type(scales) == list and all([type[i] == int for i in scales])):
            scale_list = [sequtil.get_aaindex_scale(i) for i in scales]
            scale_ids = ['aai%i' % (i) for i in scales]
            scale_names = ['amino acid index %i' % (i) for i in scales]
        else:
            raise ValueError('Incorrect scale provided: %s\n' % (str(scales)))

        return (scale_list, scale_ids, scale_names)

    def average_signal(self, scales, window, edge, feature_ids=False):
        '''
        scales: 'gg',1 ,2 ,3, ..., '1', '2', ...
        window: 5, 7, ...,
        edge: 0...100
        '''

        # fetch scales for provided scales param
        scales, scale_ids, scale_names = self._parse_scales(scales)

        if not(feature_ids):

            result = []

            for scale in scales:
                result.append(sequtil.avg_seq_signal(
                              self.protein_sequence, scale, window, edge))

            return result

        else:
            return (scale_ids, scale_names)

    def signal_peaks_area(self, scales, window, edge, threshold,
                          feature_ids=False):

        # fetch scales for provided scales param
        scales, scale_ids, scale_names = self._parse_scales(scales)

        if not(feature_ids):

            result = []

            for scale in scales:
                top, bot = sequtil.auc_seq_signal(self.protein_sequence, scale,
                                                  window, edge, threshold)
                result.append(top)
                result.append(bot)

            return result
        else:

            feat_ids = []
            feat_names = []

            for sid, sname in zip(scale_ids, scale_names):
                feat_ids.append('%stop' % (sid))
                feat_ids.append('%sbot' % (sid))
                feat_names.append('%stop' % (sname))
                feat_names.append('%sbot' % (sname))

            return (feat_ids, feat_names)

    def autocorrelation(self, ac_type, scales, lag, feature_ids=False):

        scale_list, scale_ids, scale_names = self._parse_scales(scales)

        # calculatie features
        if not(feature_ids):

            #num_feat = len(scales) * len(lags)
            result = []

            for scale in scale_list:
                result.append(sequtil.autocorrelation(ac_type,
                              self.protein_sequence, scale, lag))

            return result
        # or return feature ids and names
        else:
            return (scale_ids, scale_names)

    def length(self, feature_ids=False):
        if not(feature_ids):
            return [len(self.protein_sequence)]
        else:
            return (['len'], ['Protein length'])

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

    '''
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
    '''
    def cluster_composition(self, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_cluster_composition(self.protein_sequence)
        else:
            return (sequtil.aa_subsets, sequtil.aa_subsets)
    '''
    def five_prime_cluster_count(self, seq_length=75, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_cluster_count(self.five_prime_seq(seq_length))
        else:
            return (sequtil.aa_subsets, sequtil.aa_subsets)

    def three_prime_cluster_count(self, seq_length=75, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_cluster_count(self.three_prime_seq(seq_length))
        else:
            return (sequtil.aa_subsets, sequtil.aa_subsets)
    '''
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

    # feature calculation help functions

    def prime_seq(self, prime, length):
        '''
        This function returns the 3- or 5- prime side of the protein amino acid
        sequence. The parameter prime must be either 3 or 5 and indicates if
        the 3-prime side or the 5-prime side should be returned. The length
        indicates how long the returned prime sequence will be. If length is
        greater than or equal to the full protein sequence length, than the
        full protein sequence will be returned.

        >>> p = Protein('test')
        >>> p.set_protein_sequence('AAAAACCCCC')
        >>> p.prime_seq(5, 2)
        'AA'
        >>> p.prime_seq(3, 9)
        'AAAACCCCC'
        >>> p.prime_seq(3, 10)
        'AAAAACCCCC'
        >>> p.prime_seq(3, 11)
        'AAAAACCCCC'
        '''
        if not(prime in [3, 5]):
            raise ValueError('prime must be either 3 or 5 (int).')
        if(length < 1):
            raise ValueError('Prime length must be positive integer.')

        if(prime == 5):
            return self.protein_sequence[:length]
        else:  # prime == 3
            return self.protein_sequence[-length:]

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

    def pfam_clan_index(self, position):
        if not(self.pfam_annotations is None):
            for annotation in self.pfam_annotations:
                if not(annotation.clan is None):
                    if(position >= annotation.start_pos and
                            position <= annotation.end_pos):
                        # assuming no overlap, return the first one found
                        return annotation.clan_index
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

    def msa_num_ali_seq(self, position):
        return len(self.msa_column(position, with_gaps=True))

    def msa_num_ali_let(self, position):
        return len(self.msa_column(position, with_gaps=False))

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
        #col = self.msa_column(position, with_gaps=True)

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
