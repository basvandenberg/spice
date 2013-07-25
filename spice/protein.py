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

    # feature calculation functions

    def amino_acid_composition(self, num_segments, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_composition(self.protein_sequence, num_segments)
        else:

            feat_ids = []
            feat_names = []

            for si in xrange(1, num_segments + 1):
                for aa in sequtil.aa_unambiguous_alph:
                    feat_ids.append('%i%s' % (si, aa))
                    feat_names.append(
                        'segment %i, amino acid %s' % (si, aa))

            return (feat_ids, feat_names)

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
            return (sequtil.aa_subsets, sequtil.aa_subsets)

    def three_prime_cluster_count(self, seq_length=75, feature_ids=False):
        if not(feature_ids):
            return sequtil.aa_cluster_count(self.three_prime_seq(seq_length))
        else:
            return (sequtil.aa_subsets, sequtil.aa_subsets)

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

            scales = sequtil.get_georgiev_scales()
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

            scales = sequtil.get_georgiev_scales()
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

    def autocorrelation_mb(self, scales, lags, feature_ids=False):

        # 'parse' the scale parameter
        if(type(scales) == list and all([type[i] == int for i in scales])):
            scales = [sequtil.get_scale(i) for i in scales]
        elif(scales == 'gg'):
            # retrieve set of georgiev scales
            scales = sequtil.get_georgiev_scales()
        else:
            raise ValueError('Incorrect scale provided: %s\n'
                             % (str(scales)))

        # check lags parameter
        if not(type(lags) == list and all([type[i] == int for i in lags])):
            raise ValueError('Incorrect lags provided: %s\n'
                             % (str(lags)))

        # calculatie features
        if not(feature_ids):

            #num_feat = len(scales) * len(lags)
            result = []

            for s in scales:
                for l in lags:
                    result.append(sequtil.autocorrelation_mb(s, l))

            return result
        # or return feature ids and names
        else:

            feat_ids = []
            feat_names = []
            for s in scales:
                for l in lags:
                    feat_ids.append('acmb:%03d:%02d' % (s, l))
                    feat_names.append(
                        'Autocorrelation Moreau-Broto (scale:%03d lag:%02d' %
                        (s, l))

            return (feat_ids, feat_names)
        

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

    CLANS = [
        'CL0001', 'CL0003', 'CL0004', 'CL0005', 'CL0006', 'CL0007', 'CL0009',
        'CL0010', 'CL0011', 'CL0012', 'CL0013', 'CL0014', 'CL0015', 'CL0016', 'CL0018',
        'CL0020', 'CL0021', 'CL0022', 'CL0023', 'CL0025', 'CL0026', 'CL0027', 'CL0028',
        'CL0029', 'CL0030', 'CL0031', 'CL0032', 'CL0033', 'CL0034', 'CL0035', 'CL0036',
        'CL0037', 'CL0039', 'CL0040', 'CL0041', 'CL0042', 'CL0043', 'CL0044', 'CL0045',
        'CL0046', 'CL0047', 'CL0048', 'CL0049', 'CL0050', 'CL0051', 'CL0052', 'CL0053',
        'CL0054', 'CL0055', 'CL0056', 'CL0057', 'CL0058', 'CL0059', 'CL0060', 'CL0061',
        'CL0062', 'CL0063', 'CL0064', 'CL0065', 'CL0066', 'CL0067', 'CL0068', 'CL0069',
        'CL0070', 'CL0071', 'CL0072', 'CL0073', 'CL0074', 'CL0075', 'CL0076', 'CL0077',
        'CL0078', 'CL0079', 'CL0080', 'CL0081', 'CL0082', 'CL0083', 'CL0084', 'CL0085',
        'CL0086', 'CL0087', 'CL0088', 'CL0089', 'CL0090', 'CL0091', 'CL0092', 'CL0093',
        'CL0094', 'CL0095', 'CL0096', 'CL0097', 'CL0098', 'CL0099', 'CL0100', 'CL0101',
        'CL0103', 'CL0104', 'CL0105', 'CL0106', 'CL0107', 'CL0108', 'CL0109', 'CL0110',
        'CL0111', 'CL0112', 'CL0113', 'CL0114', 'CL0115', 'CL0116', 'CL0117', 'CL0118',
        'CL0121', 'CL0122', 'CL0123', 'CL0124', 'CL0125', 'CL0126', 'CL0127', 'CL0128',
        'CL0129', 'CL0130', 'CL0131', 'CL0132', 'CL0133', 'CL0135', 'CL0136', 'CL0137',
        'CL0139', 'CL0140', 'CL0141', 'CL0142', 'CL0143', 'CL0144', 'CL0145', 'CL0146',
        'CL0147', 'CL0148', 'CL0149', 'CL0151', 'CL0153', 'CL0154', 'CL0155', 'CL0156',
        'CL0157', 'CL0158', 'CL0159', 'CL0160', 'CL0161', 'CL0162', 'CL0163', 'CL0164',
        'CL0165', 'CL0166', 'CL0167', 'CL0168', 'CL0169', 'CL0170', 'CL0171', 'CL0172',
        'CL0173', 'CL0174', 'CL0175', 'CL0176', 'CL0177', 'CL0178', 'CL0179', 'CL0181',
        'CL0182', 'CL0183', 'CL0184', 'CL0186', 'CL0187', 'CL0188', 'CL0189', 'CL0190',
        'CL0191', 'CL0192', 'CL0193', 'CL0194', 'CL0195', 'CL0196', 'CL0197', 'CL0198',
        'CL0199', 'CL0200', 'CL0201', 'CL0202', 'CL0203', 'CL0204', 'CL0205', 'CL0206',
        'CL0207', 'CL0208', 'CL0209', 'CL0210', 'CL0212', 'CL0213', 'CL0214', 'CL0217',
        'CL0218', 'CL0219', 'CL0220', 'CL0221', 'CL0222', 'CL0223', 'CL0224', 'CL0225',
        'CL0226', 'CL0227', 'CL0228', 'CL0229', 'CL0230', 'CL0231', 'CL0232', 'CL0233',
        'CL0234', 'CL0235', 'CL0236', 'CL0237', 'CL0238', 'CL0239', 'CL0240', 'CL0241',
        'CL0242', 'CL0243', 'CL0244', 'CL0245', 'CL0246', 'CL0247', 'CL0248', 'CL0249',
        'CL0250', 'CL0251', 'CL0252', 'CL0254', 'CL0255', 'CL0256', 'CL0257', 'CL0258',
        'CL0259', 'CL0260', 'CL0261', 'CL0262', 'CL0263', 'CL0264', 'CL0265', 'CL0266',
        'CL0267', 'CL0268', 'CL0269', 'CL0270', 'CL0271', 'CL0272', 'CL0273', 'CL0274',
        'CL0275', 'CL0276', 'CL0277', 'CL0278', 'CL0279', 'CL0280', 'CL0281', 'CL0282',
        'CL0283', 'CL0284', 'CL0285', 'CL0286', 'CL0287', 'CL0288', 'CL0289', 'CL0290',
        'CL0291', 'CL0292', 'CL0293', 'CL0294', 'CL0295', 'CL0296', 'CL0297', 'CL0298',
        'CL0299', 'CL0300', 'CL0301', 'CL0302', 'CL0303', 'CL0304', 'CL0305', 'CL0306',
        'CL0307', 'CL0308', 'CL0310', 'CL0311', 'CL0312', 'CL0314', 'CL0315', 'CL0316',
        'CL0317', 'CL0318', 'CL0319', 'CL0320', 'CL0321', 'CL0322', 'CL0323', 'CL0324',
        'CL0325', 'CL0326', 'CL0327', 'CL0328', 'CL0329', 'CL0330', 'CL0331', 'CL0332',
        'CL0333', 'CL0334', 'CL0335', 'CL0336', 'CL0337', 'CL0339', 'CL0340', 'CL0341',
        'CL0342', 'CL0343', 'CL0344', 'CL0345', 'CL0346', 'CL0347', 'CL0348', 'CL0349',
        'CL0350', 'CL0351', 'CL0352', 'CL0353', 'CL0354', 'CL0355', 'CL0356', 'CL0357',
        'CL0359', 'CL0360', 'CL0361', 'CL0362', 'CL0363', 'CL0364', 'CL0365', 'CL0366',
        'CL0367', 'CL0368', 'CL0369', 'CL0370', 'CL0371', 'CL0372', 'CL0373', 'CL0374',
        'CL0375', 'CL0376', 'CL0377', 'CL0378', 'CL0379', 'CL0380', 'CL0381', 'CL0382',
        'CL0383', 'CL0384', 'CL0385', 'CL0386', 'CL0387', 'CL0388', 'CL0389', 'CL0390',
        'CL0391', 'CL0392', 'CL0393', 'CL0394', 'CL0395', 'CL0396', 'CL0397', 'CL0398',
        'CL0399', 'CL0400', 'CL0401', 'CL0402', 'CL0403', 'CL0404', 'CL0405', 'CL0406',
        'CL0407', 'CL0408', 'CL0409', 'CL0410', 'CL0411', 'CL0412', 'CL0413', 'CL0414',
        'CL0416', 'CL0417', 'CL0418', 'CL0419', 'CL0420', 'CL0421', 'CL0422', 'CL0423',
        'CL0424', 'CL0425', 'CL0426', 'CL0428', 'CL0429', 'CL0430', 'CL0431', 'CL0433',
        'CL0434', 'CL0435', 'CL0436', 'CL0437', 'CL0438', 'CL0439', 'CL0441', 'CL0442',
        'CL0444', 'CL0445', 'CL0446', 'CL0447', 'CL0448', 'CL0449', 'CL0450', 'CL0451',
        'CL0452', 'CL0453', 'CL0454', 'CL0455', 'CL0456', 'CL0457', 'CL0458', 'CL0459',
        'CL0461', 'CL0462', 'CL0464', 'CL0465', 'CL0466', 'CL0468', 'CL0469', 'CL0470',
        'CL0471', 'CL0472', 'CL0474', 'CL0475', 'CL0476', 'CL0477', 'CL0478', 'CL0479',
        'CL0480', 'CL0481', 'CL0482', 'CL0483', 'CL0484', 'CL0486', 'CL0487', 'CL0488',
        'CL0489', 'CL0490', 'CL0491', 'CL0492', 'CL0493', 'CL0494', 'CL0496', 'CL0497',
        'CL0498', 'CL0499', 'CL0500', 'CL0501', 'CL0502', 'CL0503', 'CL0504', 'CL0505',
        'CL0506', 'CL0507', 'CL0508', 'CL0509', 'CL0511', 'CL0512', 'CL0513', 'CL0515',
        'CL0516', 'CL0517', 'CL0520', 'CL0521', 'CL0522', 'CL0523', 'CL0524', 'CL0525',
        'CL0526', 'CL0527', 'CL0528', 'CL0529', 'CL0530', 'CL0531', 'CL0532', 'CL0533',
        'CL0534', 'CL0535', 'CL0536', 'CL0537', 'CL0538', 'CL0539', 'CL0540', 'CL0541',
        'CL0542', 'CL0543', 'CL0544', 'CL0545', 'CL0546', 'CL0547', 'CL0548', 'CL0549',
        'CL0550', 'CL0551', 'CL0552', 'CL0553'
    ]
    CLAN_TO_INDEX = dict(zip(CLANS, xrange(len(CLANS))))

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
        if(self.clan in self.CLAN_TO_INDEX.keys()):
            self.clan_index = self.CLAN_TO_INDEX[self.clan]
        else:
            self.clan_index = -1
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
