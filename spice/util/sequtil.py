import itertools
import numpy


###############################################################################
#
# AMINO ACID ALPHABET
#
# This section defines the amino acid alphabet and corresponding amino acid
# short and long names. The data is derived from:
# www.ebi.ac.uk/2can/biology/molecules_small_aatable.html
#
#
# IMPORTANT: order of the alphabet letters is important! do not change this.
#
###############################################################################


# unambiguos amino acid alphabet
aa_unambiguous_alph = 'ARNDCEQGHILKMFPSTWYV'
aa_unambiguous_short = [
    'ala', 'arg', 'asn', 'asp', 'cys', 'glu', 'gln', 'gly', 'his', 'ile',
    'leu', 'lys', 'met', 'phe', 'pro', 'ser', 'thr', 'trp', 'tyr', 'val'
]
aa_unambiguous_name = [
    'alanine', 'arginine', 'asparagine', 'aspartic acid', 'cysteine',
    'glutamic acid', 'glutamine', 'glycine', 'histidine', 'isoleucine',
    'leucine', 'lysine', 'methionine', 'phenylalanine', 'proline', 'serine',
    'threonine', 'tryptophan', 'tyrosine', 'valine'
]

# ambiguous amina acids
aa_ambiguous_alph = 'BJZX'
aa_ambiguous_short = ['asx', 'xle', 'xaa', 'glx']
aa_ambiguous_name = [
    'aspartic acid or asparagine', 'leucine or isoleucine',
    'unknown amino acid', 'glutamic acid or glutamine'
]

# special amino acids
aa_special_alph = 'UO'
aa_special_short = ['sec', 'pyl']
aa_special_name = ['selenocysteine', 'pyrralysine']

# stop codon translation
aa_ter_alph = '*'
aa_ter_short = ['ter']
aa_ter_name = ['terminal']

# full alphabet
aa_alph = aa_unambiguous_alph + aa_ambiguous_alph + aa_special_alph +\
    aa_ter_alph
aa_short = list(itertools.chain.from_iterable([
    aa_unambiguous_short, aa_ambiguous_short, aa_special_short, aa_ter_short
]))
aa_name = list(itertools.chain.from_iterable([
    aa_unambiguous_name, aa_ambiguous_name, aa_special_name, aa_ter_name
]))

# mapping from ambiguous letters to list of ambiguous letters
map_to_ambiguous_aa = dict(zip(list(aa_unambiguous_alph),
                           list(aa_unambiguous_alph)))
map_to_ambiguous_aa['*'] = '*'
map_to_ambiguous_aa['DN'] = 'B'
map_to_ambiguous_aa['IL'] = 'J'
map_to_ambiguous_aa['GQ'] = 'Z'


###############################################################################
#
# AMINO ACID SCALES
#
# Amino acid scales are mappings from the unambiguous amino acids to a value
# that describes some kind of property of the amino acid, such as the size or
# the hydrophobicity.
#
# Each of the rows in the following arrays is a scale, and each column contains
# the values for one amino acid. The order of the columns (amino acids) is the
# same as the order of the amino acids in aa_unambiguous_alph.
#
###############################################################################


# 19 varimax Georgiev scales
georgiev_scales_mat = numpy.array([
    [
        0.57, -2.8, -2.02, -2.46, 2.66, -3.08, -2.54, 0.15, -0.39, 3.1, 2.72,
        -3.89, 1.89, 3.12, -0.58, -1.1, -0.65, 1.89, 0.79, 2.64],
    [
        3.37, 0.31, -1.92, -0.66, -1.52, 3.45, 1.82, -3.49, 1., 0.37, 1.88,
        1.47, 3.88, 0.68, -4.33, -2.05, -1.6, -0.09, -2.62, 0.03],
    [
        -3.66, 2.84, 0.04, -0.57, -3.29, 0.05, -0.82, -2.97, -0.63, 0.26, 1.92,
        1.95, -1.57, 2.4, -0.02, -2.19, -1.39, 4.21, 4.11, -0.67],
    [
        2.34, 0.25, -0.65, 0.14, -3.77, 0.62, -1.85, 2.06, -3.49, 1.04, 5.33,
        1.17, -3.58, -0.35, -0.21, 1.36, 0.63, -2.77, -0.63, 2.34],
    [
        -1.07, 0.2, 1.61, 0.75, 2.96, -0.49, 0.09, 0.7, 0.05, -0.05, 0.08,
        0.53, -2.55, -0.88, -8.31, 1.78, 1.35, 0.72, 1.89, 0.64],
    [
        -0.4, -0.37, 2.08, 0.24, -2.23, -0., -0.6, 7.47, 0.41, -1.18, 0.09,
        0.1, 2.07, 1.62, -1.82, -3.36, -2.45, 0.86, -0.53, -2.01],
    [
        1.23, 3.81, 0.4, -5.15, 0.44, -5.66, 0.25, 0.41, 1.61, -0.21, 0.27,
        4.01, 0.84, -0.15, -0.12, 1.39, -0.65, -1.07, -1.3, -0.33],
    [
        -2.32, 0.98, -2.47, -1.17, -3.49, -0.11, 2.11, 1.62, -0.6, 3.45, -4.06,
        -0.01, 1.85, -0.41, -1.18, -1.21, 3.43, -1.66, 1.31, 3.93],
    [
        -2.01, 2.43, -0.07, 0.73, 2.22, 1.49, -1.92, -0.47, 3.55, 0.86, 0.43,
        -0.26, -2.05, 4.2, 0., -2.83, 0.34, -5.87, -0.56, -0.21],
    [
        1.31, -0.99, 7.02, 1.5, -3.78, -2.26, -1.67, -2.9, 1.52, 1.98, -1.2,
        -1.66, 0.78, 0.73, -0.66, 0.39, 0.24, -0.66, -0.95, 1.27],
    [
        -1.14, -4.9, 1.32, 1.51, 1.98, -1.62, 0.7, -0.98, -2.28, 0.89, 0.67,
        5.86, 1.53, -0.56, 0.64, -2.92, -0.53, -2.49, 1.91, 0.43],
    [
        0.19, 2.09, -2.44, 5.61, -0.43, -3.97, -0.27, -0.62, -3.12, -1.67,
        -0.29, -0.06, 2.44, 3.54, -0.92, 1.27, 1.91, -0.3, -1.26, -1.71],
    [
        1.66, -3.08, 0.37, -3.85, -1.03, 2.3, -0.99, -0.11, -1.45, -1.02,
        -2.47, 1.38, -0.26, 5.25, -0.37, 2.86, 2.66, -0.5, 1.57, -2.93],
    [
        4.39, 0.82, -0.89, 1.28, 0.93, -0.06, -1.56, 0.15, -0.77, -1.21, -4.79,
        1.78, -3.09, 1.73, 0.17, -1.88, -3.07, 1.64, 0.2, 4.22],
    [
        0.18, 1.32, 3.13, -1.98, 1.43, -0.35, 6.22, -0.53, -4.18, -1.78, 0.8,
        -2.71, -1.39, 2.14, 0.36, -2.42, 0.2, -0.72, -0.76, 1.06],
    [
        -2.6, 0.69, 0.79, 0.05, 1.45, 1.51, -0.18, 0.35, -2.91, 5.71, -1.43,
        1.62, -1.02, 1.1, 0.08, 1.75, -2.2, 1.75, -5.19, -1.31],
    [
        1.49, -2.62, -1.54, 0.9, -1.15, -2.29, 2.72, 0.3, 3.37, 1.54, 0.63,
        0.96, -4.32, 0.68, 0.16, -2.77, 3.73, 2.73, -2.56, -1.97],
    [
        0.46, -1.49, -1.71, 1.38, -1.64, -1.47, 4.35, 0.32, 1.87, 2.11, -0.24,
        -1.09, -1.34, 1.46, -0.34, 3.36, -5.46, -2.2, 2.87, -1.21],
    [
        -4.22, -2.57, -0.25, -0.03, -1.05, 0.15, 0.92, 0.05, 2.17, -4.18, 1.01,
        1.36, 0.09, 2.33, 0.04, 2.67, -0.73, 0.9, -3.43, 4.77]
])

# 10 BLOSUM62-derived Georgiev scales
georgiev_blosum_scales_mat = numpy.array([
    [
        0.077, 1.014, 1.511, 1.551, -1.084, 1.477, 1.094, 0.849, 0.716, -1.462,
        -1.406, 1.135, -0.963, -1.619, 0.883, 0.844, 0.188, -1.577, -1.142,
        -1.127],
    [
        -0.916, 0.189, 0.215, 0.005, -1.112, 0.229, 0.296, 0.174, 1.548,
        -1.126, -0.856, -0.039, -0.585, 1.007, -0.675, -0.448, -0.733, 2.281,
        1.74, -1.227],
    [
        0.526, -0.86, -0.046, 0.323, 1.562, -0.67, -0.871, 1.726, -0.802,
        -0.761, -0.879, -0.802, -0.972, -0.311, 0.382, 0.423, 0.178, 1.166,
        -0.582, -0.633],
    [
        0.004, -0.609, 1.009, 0.493, 0.814, -0.355, -0.718, 0.093, 1.547,
        0.382, -0.172, -0.849, -0.528, 0.623, -0.869, 0.317, -0.012, -1.61,
        0.747, 0.064],
    [
        0.24, 1.277, 0.12, -0.991, 1.828, -0.284, 0.5, -0.548, 0.35, -0.599,
        0.032, 0.819, 0.236, -0.549, -1.243, 0.2, 0.022, 0.122, -0.119,
        -0.596],
    [
        0.19, 0.195, 0.834, 0.01, -1.048, -0.075, -0.08, 1.186, -0.785, 0.276,
        0.344, 0.097, 0.365, 0.29, -2.023, 0.541, 0.378, 0.239, -0.475, 0.158],
    [
        0.656, 0.661, -0.033, -1.615, -0.742, -1.014, -0.442, 1.213, 0.655,
        -0.132, 0.109, 0.213, 0.062, -0.021, 0.845, 0.009, -0.304, -0.542,
        0.241, 0.014],
    [
        -0.047, 0.175, -0.57, 0.526, 0.379, 0.363, 0.202, 0.874, -0.076, 0.198,
        0.146, 0.129, 0.208, 0.098, -0.352, -0.797, -1.958, -0.398, -0.251,
        0.016],
    [
        1.357, -0.219, -1.2, -0.15, -0.121, 0.769, 0.384, 0.009, -0.186,
        -0.216, -0.436, 0.176, -0.56, 0.433, -0.421, 0.624, 0.149, -0.349,
        0.713, 0.251],
    [
        0.333, -0.52, -0.139, -0.282, -0.102, 0.298, 0.667, 0.242, 0.99, 0.207,
        -0.021, -0.85, 0.361, -1.288, -0.298, -0.129, 0.063, 0.499, -0.251,
        0.607]
])


def get_georgiev_scale(index, ambiguous=True):
    '''
    '''
    scale = dict(zip(aa_unambiguous_alph, georgiev_scales_mat[index]))
    return _get_scale(scale, ambiguous)


def get_georgiev_blosum_scale(index, ambiguous=True):
    '''
    '''
    scale = dict(zip(aa_unambiguous_alph, georgiev_blosum_scales_mat[index]))
    return _get_scale(scale, ambiguous)


def _get_scale(scale, ambiguous):
    '''
    '''
    if(ambiguous):
        scale.update(_get_non_aa_letter_dict())
    return scale


def _get_non_aa_letter_dict():
    '''
    '''
    other_letters = aa_ambiguous_alph + aa_special_alph + aa_ter_alph
    return dict(zip(other_letters, len(other_letters) * [0.0]))

georgiev_scales = [get_georgiev_scale(i) for i in xrange(19)]
georgiev_blosum_scales = [get_georgiev_blosum_scale(i) for i in xrange(10)]


###############################################################################
#
# AMINO ACID CLUSTERS
#
# 4 (www.ebi.ac.uk/2can/biology/molecules_small_aatable.html)
# 7 source: taylor85 adjusted version on the url above
# 2 wikipedia aa propensities
#
# #############################################################################


aa_subset_dict = {
    'aliphatic_hydrophobic': 'AVLIMPFW',
    'polar_uncharged': 'GSYNQC',
    'acidic': 'ED',
    'basic': 'KRH',
    'aliphatic': 'ILV',
    'aromatic': 'FYWH',
    'charged': 'HKRED',
    'polar': 'YWHKRDETCSNQ',
    'small': 'VCAGTPSDN',
    'tiny': 'AGCST',
    'helix': 'MALEK',
    'sheet': 'YFWTVI'}
aa_subsets = sorted(aa_subset_dict.keys())


###############################################################################
#
# AMINO ACID ANNOTATION SEQUENCES
#
###############################################################################


# secondary structure alphabet
ss_alph = 'CHE'
ss_short = ['col', 'hel', 'str']
ss_name = ['random coil', 'helix', 'strand']

# solvent accessibility alphabet
sa_alph = 'BE'
sa_short = ['bur', 'exp']
sa_name = ['buried', 'exposed']


###############################################################################
#
# NUCLEOTIDE ALPHABET
#
# IMPORTANT: order of the alphabet letters is important! do not change this.
#
###############################################################################


# ambiguous nucleotide alphabet
nucleotide_unambiguous_alph = 'TCAG'

# unambiguous nucleotide alphabet
nucleotide_ambiguous_alph = 'MRWSYKVHDBN'

# full nucleotide alphabet
nucleotide_alph = nucleotide_unambiguous_alph + nucleotide_ambiguous_alph

# mapping from ambiguous alphabet to list of unambiguous nucleotides
map_to_unambiguous_nucleotides = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'U': 'T',
    'W': 'AT', 'S': 'GC', 'M': 'AC', 'K': 'GT', 'R': 'AG', 'Y': 'CT',
    'B': 'CGT', 'D': 'AGT', 'H': 'ACT', 'V': 'ACG',
    'N': 'ACGT'
}


###############################################################################
#
# CODON ALPHABET
#
# IMPORTANT: order of the alphabet is important! do not change this.
#
###############################################################################


# unambiguous codon alphabet
codons_unambiguous = [a+b+c for a in nucleotide_unambiguous_alph
                      for b in nucleotide_unambiguous_alph
                      for c in nucleotide_unambiguous_alph]

# full codon alphabet
codons = [a+b+c for a in nucleotide_alph for b in nucleotide_alph
          for c in nucleotide_alph]


###############################################################################
#
# CODON TRANSLATION UTILS
#
###############################################################################


# amino acids corresponding to codons in codons_unambiguous
codon_aas = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'

# unambiguous codon table mapping (codon --> amino acid)
codon_table_unambiguous = dict(zip(codons_unambiguous, codon_aas))


def unambiguous_codons(codon):
    '''
    This function returns all possible unambiguous codons for the provided,
    possibly ambiguous, codon.

    Args:
        codon (str): A possibly ambiguous codon.
    Raises:
        TODO

    If codon is an unambiguous codon, the list with only this codon is
    returned.

    >>> unambiguous_codons('GCG')
    ['GCG']

    If codon is an ambiguous codon, the list with all possible unambiguous
    codons for this particular codon is returned.

    >>> unambiguous_codons('ATN')
    ['ATA', 'ATC', 'ATG', 'ATT']
    '''
    return [x + y + z for x in map_to_unambiguous_nucleotides[codon[0]]
            for y in map_to_unambiguous_nucleotides[codon[1]]
            for z in map_to_unambiguous_nucleotides[codon[2]]]


def unambiguous_aas(codon):
    '''
    This function returns the amino acids for which the provided (possibly
    ambiguous) codon encodes.

    Args:
        codon (str): A possibly ambiguous codon.
    Raises:
        TODO

    If the provided codon is ambiguous, the amino acid letter for which this
    codon encodes is returned.

    >>> unambiguous_aas('ATG')
    'M'

    If the provided codon is unambiguous, a string with all possible amino
    acids that can be encoded by this codon is returned. Thes returned string
    is sorted alphabetically.

    >>> unambiguous_aas('ATN')
    'IM'
    >>> unambiguous_aas('ANN')
    'IKMNRST'
    >>> unambiguous_aas('NNN')
    '*ACDEFGHIKLMNPQRSTVWY'

    '''
    return ''.join(sorted(set([codon_table_unambiguous[c]
                   for c in unambiguous_codons(codon)])))


def ambiguous_aa(codon):
    '''
    This function returns the posibbly ambiguous amino acid that is encoded by
    the provided codon. In all cases, this method returns only one letter.

    Args:
        codon (str):
    Raises:
        TODO

    If the codon is unambiguous, the corresponding unambiguous amino acid is
    returned.

    >>> ambiguous_aa('ATG')
    'M'

    If the codon is ambiguous but encodes for a single amino acid, this amino
    acid is returned.

    >>> ambiguous_aa('TCN')
    'S'

    If the codon is ambiguous and does not encode for a single amino acid, the
    ambiguous amino acid that best represents the set of unambiguous amino
    acids that can be encoded by the ambiguous codon is returned.

    >>> ambiguous_aa('MTT')
    'J'

    In most cases this will be an X, which encodes for any amino acid.

    >>> ambiguous_aa('ATN')
    'X'
    '''
    sorted_unamb_aas = unambiguous_aas(codon)
    return map_to_ambiguous_aa.get(sorted_unamb_aas, 'X')

# ambiguous codon table mapping
codon_table = dict(zip(codons, [ambiguous_aa(c) for c in codons]))

'''
# number of codons per amino acid
ncodon_per_aa = numpy.array([codon_aas.count(l) for l in codon_aas])
'''


###############################################################################
#
# SEQUENCE OPERATIONS
#
###############################################################################


def translate(orf):
    '''
    This function translates a (possibly ambiguous) ORF nucleotide sequence
    into an amino acid protein sequence.

    Args:
        orf (str): The open reading frame (nucleotide) sequence.

    Returns:
        str The translated protein (amino acid) sequence.

    Raises:
        ValueError: if the orf length is not a multiple of 3.
        ValueError: if orf is not an (possibly ambiguous) nucleotide sequence.

    Translation of a random ORF sequence part can be done with:

    >>> translate('ATGTTTAGTAACAGACTACCACCTCCAAAA')
    'MFSNRLPPPK'

    Ambiguous codons will be translated to corresponding (possibly ambiguous)
    amino acids.

    >>> translate('ATGMTTAGTAACAGACTACCACCTCCAAAA')
    'MJSNRLPPPK'
    '''
    if not(len(orf) % 3 == 0):
        raise ValueError('ORF sequence length is not a multiple of 3.')
    if not(is_nucleotide_sequence(orf)):
        raise ValueError('ORF sequence is not a nucleotide sequence.')

    return ''.join([codon_table[orf[i:i+3]] for i in xrange(0, len(orf), 3)])


###############################################################################
#
# GLOBAL SEQUENCE FEATURES
#
###############################################################################


def seq_count(seq, alph):
    '''
    This function counts letter occurance in seq for each letter in alph.

    Args:
        seq (str): The sequence of which the letters will be counted.
        alph (str): The letters that will be counted in seq

    Returns:
        numpy.array List with letter counts in the order of alph.

    >>> seq_count('AABBCBBACB', 'ABC')
    array([3, 5, 2])
    >>> seq_count('', 'ABC')
    array([0, 0, 0])
    >>> seq_count('ABC', '')
    array([], dtype=int64)
    '''
    return numpy.array([seq.count(l) for l in alph], dtype=int)


def seq_composition(seq, alph):
    '''
    This function returns the letter composition of seq for the letters in
    alph.

    Args:
        seq (str):
        alph (str):

    Raises:
        ValueError: if the sequence is empty.

    If seq contains only letters that are in alph, than the returned
    list of floats adds to one. Otherwise the sum of the numbers is between
    0.0 and 1.0

    >>> seq_composition('AABBCBBACB', 'ABC')
    array([ 0.3,  0.5,  0.2])
    >>> sum(seq_composition('AABBCBBACB', 'ABC'))
    1.0
    >>> seq_composition('AABBCBBACB', 'AB')
    array([ 0.3,  0.5])
    >>> seq_composition('AABBCBBACB', 'AD')
    array([ 0.3,  0. ])
    >>> seq_composition('AAAAAAAAAA', 'A')
    array([ 1.])
    >>> seq_composition('AAAAAAAAAA', '')
    array([], dtype=float64)
    '''
    if(len(seq) == 0):
        raise ValueError('Cannot calculate composition of empty sequence.')
    return seq_count(seq, alph) / float(len(seq))


def state_subseq(seq, state_seq, state_letter):
    '''
    This function returns the returns those parts of seq where the
    state_seq letter is equal to state_letter. The subparts are glued together
    and returned as a single string.

    >>> state_subseq('ABCDEFGHIJKLMNO', 'AAABBBCCCAAABBB', 'B')
    'DEFMNO'
    >>> state_subseq('ABCDEFGHIJKLMNO', 'AAABBBCCCAAABBB', 'D')
    ''
    >>> state_subseq('ABCDEFGHIJKLMNO', 'AAAAAACCCAAACCC', 'B')
    ''
    >>> state_subseq('ABCDEFGHIJKLMNO', 'AAAAAACCCAAACCC', '')
    ''
    >>> state_subseq('', '', 'A')
    ''
    '''
    if not(len(seq) == len(state_seq)):
        raise ValueError('The state_seq should have the same length as seq.')

    return ''.join([l if state_seq[i] == state_letter else ''
                   for i, l in enumerate(seq)])


def state_subseq_composition(seq, state_seq, seq_alph, state_alph):
    '''
    This function returns the seq_alph composition of seq, but only for the
    parts where the state_seq has a letter that is in state_alph.

    >>> s = 'SSSSSTTTTTSSSS'
    >>> t = 'AAABBCCAAAAAAA'
    >>> comp = state_subseq_composition(s, t, 'ST', 'A')
    >>> print(round(comp[0], 1))
    0.7
    >>> print(round(comp[1], 1))
    0.3
    '''
    result = []
    # TODO is extend correct??? Now all composition are added to the same list
    for l in state_alph:
        result.extend(seq_composition(state_subseq(seq, state_seq, l),
                      seq_alph))
    return result


def aa_count(protein):
    '''
    This function returns the (unambiguous) amino acid count of the provided
    protein sequence.
    '''
    return seq_count(protein, aa_unambiguous_alph)


def aa_composition(protein):
    '''
    This function returns the amino acid composition of the provided protein
    sequence. Only unambiguous amino acids are considered, therefore the
    result does not need to sum to 1.0.
    '''
    return seq_composition(protein, aa_unambiguous_alph)


def ss_composition(protein_ss):
    '''
    This function returns the secondary structure composition of the provided
    protein secondary structure sequence.
    '''
    return seq_composition(protein_ss, ss_alph)


def sa_composition(protein_sa):
    '''
    This function returns the solvent accessibility compositions of the
    provided protein solvent accessibility sequence.
    '''
    return seq_composition(protein_sa, sa_alph)


def ss_aa_composition(protein, ss):
    '''
    This function returns the amino acid composition per (combined) secondairy
    structure region.

    Args:
        protein (str): The protein amino acid sequence.
        ss (str): The corresponding secondary structure sequence.
    Raises:
        TODO

    The sequence parts that are in one type of secondairy structure, i.e. in a
    helix, are combined into one string and the amino acid composition of this
    sequence is returned. This is also done for the strand and random coil
    regions.
    '''
    return state_subseq_composition(protein, ss, aa_unambiguous_alph, ss_alph)


def sa_aa_composition(protein, sa):
    '''
    This function returns the amino acid composition of both the buried and the
    exposed parts of the protein.

    Args:
        protein (str): The protein amino acid sequence.
        sa (str): The solvent accessibility sequence.

    The buried and exposed parts are combined into separate sequences and the
    amino acid composition of both of these sequences is returned.
    '''
    return state_subseq_composition(protein, sa, aa_unambiguous_alph, sa_alph)

'''
def aa_cluster_count(protein):
    counts = dict(zip(aa_unambiguous_alph, aa_count(protein)))
    return numpy.array([sum([comp[l] for l in aa_subset_dict[subset]])
            for subset in aa_subsets])
'''


def aa_cluster_composition(protein):
    '''
    This function returns the protein sequence composition of the different
    defined amino acid clusters.
    '''
    comp = dict(zip(aa_unambiguous_alph, aa_composition(protein)))
    return numpy.array([sum([comp[l] for l in aa_subset_dict[subset]])
                       for subset in aa_subsets])


def window_seq(seq, window_size, overlapping=False):
    '''
    This function returns a chopped version of seq, in which it is chopped in
    subsequences of length window_size. By default, non-overlapping
    subsequences are returned, if overlapping is set to True, overlapping sub-
    sequences are returned.

    IMPORTANT: If the length of the sequence is not a multiple of the window
    size, the last letters are NOT returned.

    >>> s = 'ACCACCAAAA'
    >>> window_seq(s, 3)
    ['ACC', 'ACC', 'AAA']
    >>> window_seq(s, 3, overlapping=True)
    ['ACC', 'CCA', 'CAC', 'ACC', 'CCA', 'CAA', 'AAA', 'AAA']
    >>> window_seq(s, 1)
    ['A', 'C', 'C', 'A', 'C', 'C', 'A', 'A', 'A', 'A']
    >>> window_seq(s, 10)
    ['ACCACCAAAA']
    >>> window_seq(s, 11)
    []
    '''

    if(window_size < 2):
        return list(seq)
    else:
        start = 0
        stop = len(seq) - window_size + 1
        step = window_size
        if(overlapping):
            step = 1
        return [seq[i:i + window_size] for i in range(start, stop, step)]


filter_cache = {}


def convolution_filter(window=9, edge=0.0):
    '''
    This function returns a triangular convolution filter. The filter values
    add up to 1.0.

    Args:
        window (int): The width of the filter
        edge (float): The weight of the edges of the window [0.0, 1.0]
    Raises:
        ValueError: if the window is not an uneven number.
        ValueError: if the window is too small, smaller than 3.
        ValueError: if the edge parameter is out of range [0.0, 1.0].

    >>> convolution_filter()
    array([ 0.    ,  0.0625,  0.125 ,  0.1875,  0.25  ,  0.1875,  0.125 ,
            0.0625,  0.    ])
    >>> convolution_filter(window=3, edge=0.333333333333)
    array([ 0.2,  0.6,  0.2])
    '''

    if((window, edge) in filter_cache.keys()):
        return filter_cache[(window, edge)]

    if(window % 2 == 0):
        raise ValueError('Window must be an uneven number.')
    if(window < 3):
        raise ValueError('Window must be 3 or larger.')
    if(edge < 0.0 or edge > 1.0):
        raise ValueError('The edge parameter must be in the range 0.0 to 1.0.')

    if(edge == 1.0):
        result = numpy.ones(window) / window
        filter_cache[(window, edge)] = result
        return result
    else:
        result = numpy.ones(window)
        num = window / 2
        forw = numpy.linspace(edge, 1.0, num, endpoint=False)
        result[:num] = forw
        result[-num:] = forw[::-1]
        result = result / result.sum()
        filter_cache[(window, edge)] = result
        return result


def seq_signal_raw(sequence, scale):
    '''
    This function maps the sequence to a raw value signal using the provided
    alphabet scale.

    Args:
        sequence (str): The sequence to be mapped.
        scale (dict(str --> float)): A mapping from each letter in the sequence
                                     alphabet to a float value.
    Raises:
        KeyError: if sequence contains letters that have no scale value.

    >>> s = 'AABBAA'
    >>> sc = {'A': -1.0, 'B': 1.0}
    >>> seq_signal_raw(s, sc)
    [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    '''
    return [scale[letter] for letter in sequence]


def seq_signal(sequence, scale, window=9, edge=0.0):
    '''
    This function returns a smoothed sequence signal using the provided letter
    to value scale and a triangular smoothing filter with the provided window
    width and edge weights.

    Args:
        sequence (str):
        scale (float):
        window (int):
        edge (float):
    Raises:
        ValueError: if window is too small or large.

    >>> s = 'AABBAA'
    >>> sc = {'A': -1.0, 'B': 1.0}
    >>> seq_signal(s, sc, window=3, edge=1.0)
    array([-0.33333333,  0.33333333,  0.33333333, -0.33333333])
    '''

    if(window > len(sequence) or window < 1):
        raise ValueError('1 <= window <= sequence length.')

    # obtain raw signal
    signal = seq_signal_raw(sequence, scale)

    # return the raw signal if window size is one
    if(window == 1):
        return signal

    # otherwise return convolved signal
    else:
        conv = convolution_filter(window, edge)
        return numpy.convolve(signal, conv, 'valid')


def avg_seq_signal(sequence, scale, window=9, edge=0.0):
    '''
    This function returns the average value of the smoothed sequence signal
    that is constructed using scale and a triangular filter with width window
    and edge weights.
    '''
    sig = seq_signal(sequence, scale, window, edge)
    return sum(sig) / len(sequence)


def auc_seq_signal(sequence, scale, window=9, edge=0.0, threshold=1.0):
    '''
    This function returns sequence signal area above and underneeth the
    specified threshold, normalized by the sequence length.

    The most basic area estimation is used.

    >>> s = 'AAABBBCCCAAA'
    >>> sc = {'A': 0.0, 'B': 1.0, 'C': -1.0}
    >>> auc_seq_signal(s, sc, window=3, edge=0.0, threshold=0.5)
    (0.125, 0.125)
    '''

    area_above = 0.0
    area_below = 0.0

    sig = seq_signal(sequence, scale, window, edge)
    for value in sig:
        if value > threshold:
            area_above += value - threshold
        elif value < -1.0 * threshold:
            area_below += -1.0 * value - threshold
        else:
            pass

    return (area_above / len(sequence), area_below / len(sequence))


def codon_count(orf):
    '''
    This function returns the codon counts in the orf sequence.

    Args:
        orf (str): Open reading frame (nucleotide) sequence
    '''
    wseq = window_seq(orf, 3, overlapping=False)
    return numpy.array([wseq.count(c) for c in codons_unambiguous])


def codon_composition(orf):
    '''
    This function returns the codon composition of the provided orf sequence.

    Args:
        orf (str): Open reading frame (nucleotide) sequence
    '''
    return codon_count(orf) / float(len(orf))


def codon_usage(orf):
    '''

    '''

    # TODO leave out codons that encode for only one amino acid?
    # What about start and stop codon?

    # count amino acids for translated orf sequence (dict: aa --> count)
    aa_c_dict = dict(zip(aa_unambiguous_alph, aa_count(translate(orf))))

    # turn into list with amino acid count per codon
    # change 0 to 1, to prevent / 0 (answer will still always be 0 (0/1))
    aa_c = numpy.array([float(aa_c_dict.get(a, 0)) if aa_c_dict.get(a, 0) > 0
                       else 1.0 for a in codon_aas])

    # get the codon counts
    codon_c = codon_count(orf)

    # divide codon count by corresponding amino acid count
    return codon_c / aa_c


###############################################################################
#
# SEQUENCE CHECKS
#
###############################################################################


def is_amino_acid_sequence(sequence):
    return set(sequence).issubset(set(aa_alph))


def is_unambiguous_amino_acid_sequence(sequence):
    return set(sequence).issubset(set(aa_unambiguous_alph))


def is_nucleotide_sequence(sequence):
    return set(sequence).issubset(set(nucleotide_alph))


def is_unambiguous_nucleotide_sequence(sequence):
    return set(sequence).issubset(set(nucleotide_unambiguous_alph))


def is_sec_struct_sequence(sequence):
    return set(sequence).issubset(set(ss_alph))


def is_solv_access_sequence(sequence):
    return set(sequence).issubset(set(sa_alph))


def probably_nucleotide(sequence):
    pass


###############################################################################
#
# Sequence properties TODO order and document this...
#
###############################################################################

'''
# human mutations: key is from, values are occuring to mutations
non_zero_mutation_counts = {
    'A':  'DEGPSTV',
    'R':  'CQGHILKMPSTW',
    'N':  'DHIKSTY',
    'D':  'ANEGHYV',
    'C':  'RGFSWY',
    'E':  'ADQGKV',
    'Q':  'REHLKP',
    'G':  'ARDCESWV',
    'H':  'RNDQLPY',
    'I':  'RNLKMFSTV',
    'L':  'RQHIMFPSWV',
    'K':  'RNEQIMT',
    'M':  'RILKTV',
    'F':  'CILSYV',
    'P':  'ARQHLST',
    'S':  'ARNCGILFPTWY',
    'T':  'ARNIKMPS',
    'W':  'RCGLS',
    'Y':  'NDCHFS',
    'V':  'ADEGILMF'
}
'''


def hamming_distance(s0, s1):
    assert(len(s0) == len(s1))
    return sum([not s0[i] == s1[i] for i in range(len(s0))])


def dist_one_codons(codon):
    return [c for c in codons_unambiguous if hamming_distance(codon, c) == 1]


def dist_one_amino_acids(codon):
    codons = dist_one_codons(codon)
    return sorted(set([codon_table_unambiguous[c] for c in codons]))


def aa_substitutions():
    tuples = [(l0, l1) for l0 in aa_unambiguous_alph
              for l1 in aa_unambiguous_alph if not l0 == l1]
    return tuples


def mutations():
    return [(l0, l1) for l0 in nucleotide_unambiguous_alph
            for l1 in nucleotide_unambiguous_alph if not l0 == l1]


def codon_mutations():
    result = []
    for codon in codons_unambiguous:
        result.extend(codon_mutation(codon))
    return result


def codon_mutation(codon):
    assert(codon in codons_unambiguous)
    return [(codon, c1) for c1 in codons_unambiguous
            if hamming_distance(codon, c1) == 1]


def single_mutation_aa_substitutions():
    return [(codon_table_unambiguous[c[0]], codon_table_unambiguous[c[1]])
            for c in codon_mutations()]


def possible_single_mutation_aa_substitutions():
    return sorted(set([s for s in single_mutation_aa_substitutions()
                  if not(s[0] == s[1] or (s[0] == '*' or s[1] == '*'))]))


def impossible_single_mutation_aa_substitutions():
    return sorted(set(aa_substitutions()) -
                  set(possible_single_mutation_aa_substitutions()))


def single_mutation_aa_substitution_stats():
    subs = single_mutation_aa_substitutions()
    no_sub = 0
    stop_sub = 0
    aa_subs = []
    for s in subs:
        if(s[0] == s[1]):
            no_sub += 1
        elif(s[0] == '*' or s[1] == '*'):
            stop_sub += 1
        else:
            aa_subs.append(s)
    print ''
    print 'No substitution: %i' % (no_sub)
    print 'Stop codon substitution: %i' % (stop_sub)
    print 'Amino acid substitution: %i' % (len(aa_subs))
    print 'TOTAL: %i' % (len(subs))
    print ''
    possible_subs = set(aa_subs)
    impossible_subs = set(aa_substitutions()) - possible_subs
    print 'Possible amino acid substitutions: %i' % (len(possible_subs))
    print 'Impossible amino acid substitutions: %i' % (len(impossible_subs))
    print 'TOTAL: %i' % (len(aa_substitutions()))
    #print impossible_subs
    print ''
