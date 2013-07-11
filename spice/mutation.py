import numpy
import scipy
import prody

from util import sequtil


class MissenseMutation(object):

    def __init__(self, protein, position, aa_from, aa_to, label, aa_pep,
                 aa_pep_i, codons, codon_fr, codons_to, pdb_id, pdb_resnum):

        # check if the mutation corresponds to the protein sequence
        assert(protein.protein_sequence[position - 1] == aa_from)

        # store all attributes
        self.protein = protein
        self.position = position
        self.aa_from = aa_from
        self.aa_to = aa_to

        self.label = label  # int
        self.aa_pep = aa_pep
        self.aa_pep_i = aa_pep_i
        self.codons = codons
        self.codon_fr = codon_fr
        self.codons_to = codons_to

        self.pdb_id = pdb_id  # None if not available
        if(self.pdb_id):
            self.pdb_chain = pdb_id.split('_')[-1]
        else:
            self.pdb_chain = None
        self.pdb_resnum = pdb_resnum  # -1 if not available

        # create unique id
        self.mid = '_'.join([protein.pid, str(position), aa_from, aa_to])

        # add this mutation to the protein (NOTE: two direction reference)
        self.protein.add_missense_mutation(self)

    '''
    TODO use proper getter setter methods...

    TODO get rid of the pointer to protein so that a mutations can live on
    itself and we can use a parse method to create a mutation from a string
    out of a mutation file...

    def single_line_str(self):
        return '%i\t%i\t%s\t%s\t%s\t%.1f\t%e\t%s\t%s' % (self.start_pos,
                self.end_pos, self.hmm_acc, self.hmm_name, self.type_,
                self.bit_score, self.e_value, self.clan, self.active_residues)

    @classmethod
    def parse(self, s):

        tokens = s.split()

        uni_id = int(tokens[0])
        pos = int(tokens[1])
        fr = tokens[2]
        to = tokens[3]
        label = int(tokens[4])
        pep = tokens[5]
        pep_i = tokens[6]
        codons = tokens[7]
        fr_codon = tokens[8]
        to_codons = tokens[9].split(',')
        pdb_id = tokens[10]
        pdb_resnum = tokens[11]


        return self(...)
    '''

    def tuple_representation(self):
        return (self.protein.pid, self.position, self.aa_from, self.aa_to,
                self.label, self.aa_pep, self.aa_pep_i, self.codons,
                self.codon_fr, self.codons_to, self.pdb_id, self.pdb_resnum)

    # feature calculation functions

    def mutation_vector(self, feature_ids=False):

        alph = sequtil.aa_unambiguous_alph

        if not(feature_ids):
            vector_repr = len(alph) * [0]
            vector_repr[alph.index(self.aa_from)] = -1
            vector_repr[alph.index(self.aa_to)] = 1
            return vector_repr
        else:
            names = sequtil.aa_unambiguous_name
            return (list(alph), names)

    def georgiev_signal_diff(self, feature_ids=False):

        num_scales = 19

        if not(feature_ids):
            feat_vec = numpy.zeros(num_scales)
            for index in xrange(num_scales):
                scale = sequtil.georgiev_scales[index]
                feat_vec[index] = self.mutation_signal_distance(scale)
            return feat_vec
        else:
            ids = ['%i' % (i) for i in xrange(num_scales)]
            names = ['Georgiev %i signal difference' % (i)
                     for i in xrange(num_scales)]
            return (ids, names)

    def georgiev_blosum_signal_diff(self, feature_ids=False):
        '''
        '''
        num_scales = 10

        if not(feature_ids):
            feat_vec = numpy.zeros(num_scales)
            for index in xrange(num_scales):
                scale = sequtil.georgiev_blosum_scales[index]
                feat_vec[index] = self.mutation_signal_distance(scale)
            return feat_vec
        else:
            ids = ['%i' % (i) for i in xrange(num_scales)]
            names = ['Georgiev Blosum62 %i signal difference' % (i)
                     for i in xrange(num_scales)]
            return (ids, names)

    def georgiev_signal_auc(self, env_window=21, sig_window=9, edge=1.0,
                            threshold=1.5, below_threshold=False,
                            feature_ids=False):

        num_scales = 19

        if not(feature_ids):
            feat_vec = numpy.zeros(num_scales)

            for index in xrange(num_scales):
                scale = sequtil.georgiev_scales[index]
                auc = self.environment_signal_peak_area(
                    env_window, scale, sig_window, edge, threshold,
                    below_threshold)
                # anscombe transform (~poissos --> ~normal)
                feat_vec[index] = 2 * numpy.sqrt(auc + (3.0 / 8.0))
            return feat_vec
        else:
            ids = ['%i' % (i) for i in xrange(num_scales)]
            names = ['Georgiev %i signal ew%i sw%i e%.2f th%.2f' %
                     (i, env_window, sig_window, edge, threshold)
                     for i in xrange(num_scales)]
            return (ids, names)

    def georgiev_blosum_signal_auc(self, env_window=21, sig_window=9, edge=1.0,
                                   threshold=1.5, below_threshold=False,
                                   feature_ids=False):

        num_scales = 10

        if not(feature_ids):
            feat_vec = numpy.zeros(num_scales)

            for index in xrange(num_scales):
                scale = sequtil.georgiev_blosum_scales[index]
                auc = self.environment_signal_peak_area(
                    env_window, scale, sig_window, edge, threshold,
                    below_threshold)
                # anscombe transform (~poissos --> ~normal)
                feat_vec[index] = 2 * numpy.sqrt(auc + (3.0 / 8.0))
            return feat_vec
        else:
            ids = ['%i' % (i) for i in xrange(num_scales)]
            names = ['Georgiev blosum %i signal ew%i sw%i e%.2f th%.2f' %
                     (i, env_window, sig_window, edge, threshold)
                     for i in xrange(num_scales)]
            return (ids, names)

    def backbone_angles(self, feature_ids=False):

        if not(feature_ids):

            # default set to zeros... TODO check
            feat_vec = numpy.zeros(3)

            if not(self.pdb_resnum == -1):

                # obtain residue object from prody structure
                struct = self.protein.get_structure()
                p = struct.select('protein')
                hv = p.getHierView()
                r = hv.getResidue(self.pdb_chain, self.pdb_resnum)

                # HACK for positions with letter...
                # TODO figure out what they are and how to handle these
                try_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                try_index = 0
                while(r is None):
                    r = hv.getResidue(self.pdb_chain, self.pdb_resnum,
                                      icode=try_letters[try_index])
                    try_index += 1

                # measure angles, value error is raised if neighbor residue
                # is not available, set default value in that case...
                # TODO check if default values makes sense...
                try:
                    omega = prody.measure.calcOmega(r)
                except ValueError:
                    omega = 0.0
                try:
                    phi = prody.measure.calcPhi(r)
                except ValueError:
                    phi = 0.0
                try:
                    psi = prody.measure.calcPsi(r)
                except ValueError:
                    psi = 0.0

                feat_vec[0] = omega
                feat_vec[1] = phi
                feat_vec[2] = psi

            return feat_vec

        else:
            angles = ['ome', 'phi', 'psi']
            names = ['omega', 'phi', 'psi']
            return (angles, names)

    def backbone_bond_distances(self, feature_ids=False):
        pass

    def solv_access(self, feature_ids=False):

        # the list gives the rasa's per residue in the pdb, so they need to
        # mapped to the correct sequence indices...
        # TODO this is not efficient... it would also be better to put this
        # stuff in the Protein class...

        if not(feature_ids):

            # default set to 1.0...
            feat_vec = numpy.ones(1)

            if not(self.pdb_resnum == -1):

                rasa_list = self.protein.get_rasa()

                # pfff, erg omslagtig allemaal dit...

                # obtain residue object from prody structure
                struct = self.protein.get_structure()
                p = struct.select('protein')
                hv = p.getHierView()

                # obtain the list of all residues, ignoring alternative res
                residues = [r for r in hv.iterResidues() if r.getIcode() == '']

                # HACK, for now we ignore the exeption occuring because of
                # insertion codes in the pdb files
                #assert(len(residues) == len(rasa_list))
                if(len(residues) == len(rasa_list)):

                    # obtain the mutated residue
                    r = hv.getResidue(self.pdb_chain, self.pdb_resnum)
                    # HACK for positions with letter...
                    # TODO figure out what they are and how to handle these
                    try_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                    try_index = 0
                    while(r is None):
                        r = hv.getResidue(self.pdb_chain, self.pdb_resnum,
                                          icode=try_letters[try_index])
                        try_index += 1

                    # get the pdb index of the mutated residue
                    pdb_index = residues.index(r)
                    rasa = rasa_list[pdb_index]

                    feat_vec[0] = rasa

                else:
                    #print
                    #print self.protein.pid
                    #print len(rasa_list)
                    #print len(residues)
                    pass

            return feat_vec

        else:
            return (['rasa'], ['rasa'])

    def atom_count(self, min_dist=1, max_dist=3, feature_ids=False):
        '''
        This function counts atoms at a certain distance from the substituted
        residue.
        '''

        atoms = ['C', 'N', 'O', 'S']  # TODO add more

        if not(feature_ids):

            # default to counts zero
            feat_vec = numpy.zeros(len(atoms))

            if not(self.pdb_resnum == -1):

                # counts min_dist and max_dist
                inner_counts = dict(zip(atoms, [0] * len(atoms)))
                outer_counts = dict(zip(atoms, [0] * len(atoms)))

                # get structure and select protein
                struct = self.protein.get_structure()
                p = struct.select('protein')

                # select atoms within min_dist
                sel_min = p.select('exwithini of resnum %i' %
                                   (min_dist, self.pdb_resnum))

                # count atoms
                if(sel_min):
                    for item in sel_min.getElements():
                        try:
                            inner_counts[item] += 1
                        except KeyError:
                            #print item
                            pass

                # select atoms within max_dist
                sel_max = p.select('exwithini of resnum %i' %
                                   (max_dist, self.pdb_resnum))

                if(sel_max):
                    for item in sel_max.getElements():
                        try:
                            outer_counts[item] += 1
                        except KeyError:
                            #print item
                            pass

                for index, atom in enumerate(atoms):
                    feat_vec[index] = outer_counts[atom] - inner_counts[atom]

            return feat_vec
        else:
            names = ['carbon', 'nitrogen', 'oxygen', 'sulfur']
            return (atoms, names)

    def seq_env_aa_count(self, window=19, feature_ids=False):

        alph = sequtil.aa_unambiguous_alph

        if not(feature_ids):
            if(window == 19):
                subseq = self.aa_pep
            else:
                subseq = self.seq_env(window)
            return sequtil.aa_count(subseq)
        else:
            names = sequtil.aa_unambiguous_name
            return (list(alph), names)

    def msa_based(self, feature_ids=False):
        if not(feature_ids):
            variability = self.protein.msa_variability[self.position - 1]
            cov = self.protein.msa_coverage[self.position - 1]
            var = len(variability)
            rank = self.protein.msa_residue_rank[self.position - 1]
            toinvar = 1.0 if self.aa_to in variability else 0.0
            return [cov, var, rank, toinvar]
        else:
            ids = ['cov', 'var', 'ran', 'toinvar']
            names = ['msa coverage', 'msa variability', 'msa rank',
                     'to residue in msa variability']
            return (ids, names)

    def msa(self, feature_ids=False):
        if not(feature_ids):
            fwtg = self.protein.msa_fraction(self.position, self.aa_from, True)
            fmut = self.protein.msa_fraction(self.position, self.aa_to, False)
            #ent = self.protein.msa_entropy21(self.position, False)
            return [fwtg, fmut]
        else:
            ids = ['fwtg', 'fmut']
            names = ['msa wt frequency with gaps', 'msa mutant frequency']
            return (ids, names)

    def msa_scale_diff(self, feature_ids=False):

        num_scales = 19

        if not(feature_ids):
            feat_vec = numpy.zeros(num_scales)
            for index in xrange(num_scales):
                scale = sequtil.georgiev_scales[index]
                feat_vec[index] = self.min_signal_dist_to_msa(scale)
            return feat_vec

        else:
            ids = ['%i' % (i) for i in xrange(num_scales)]
            names = ['Georgiev %i signal dist. to msa variability' % (i)
                     for i in xrange(num_scales)]
            return (ids, names)

    def pfam_annotation(self, feature_ids=False):

        if not(feature_ids):

            pf_fam = self.pfam_family()
            pf_dom = self.pfam_domain()
            #pf_rep = self.pfam_repeat()
            pf_cla = self.pfam_clan()
            #pf_act = self.pfam_active_residue()
            pf_cla_i = self.pfam_clan_index()

            num_features = 3
            feat_vec = numpy.zeros(num_features)

            feat_vec[0] = 0 if pf_fam is None else 1
            feat_vec[1] = 0 if pf_dom is None else 1
            #feat_vec[2] = 0 if pf_rep is None else 1
            feat_vec[2] = 0 if pf_cla is None else 1
            #feat_vec[4] = 1 if pf_act else 0
            feat_vec[3] = -1 if pf_cla_i in None else pf_cla_i

            return feat_vec

        else:
            #ids = ['fam', 'dom', 'rep', 'cla', 'act']
            #names = ['pfam family', 'pfam domain', 'pfam repeat', 'pfam clan',
            #        'pfam active residue']
            ids = ['fam', 'dom', 'cla', 'cli']
            names = ['pfam family', 'pfam domain', 'pfam clan',
                     'pfam clan index']
            return (ids, names)

    def from_codon_vector(self, feature_ids=False):
        '''
        From codon is 1, rest of codons 0.
        '''

        codons = sequtil.codons_unambiguous

        if not(feature_ids):
            vector_repr = len(codons) * [0]
            vector_repr[codons.index(self.codon_fr)] = 1
            return vector_repr

        else:
            ids = [c.lower() for c in codons]
            names = ['%s (%s)' % (c, sequtil.codon_table_unambiguous[c])
                     for c in codons]
            return (ids, names)

    def seq_env_codon_count(self, feature_ids=False):
        '''
        Counts codons of the stored codon region around the mutation.
        '''

        codons = sequtil.codons_unambiguous

        if not(feature_ids):
            return [self.codons.count(c) for c in codons]
        else:
            ids = [c.lower() for c in codons]
            names = ['%s (%s)' % (c, sequtil.codon_table_unambiguous[c])
                     for c in codons]
            return (ids, names)

    def residue_flexibility(self, feature_ids=False):

        if not(feature_ids):

            # residue flexibility
            seq_i = self.position - 1
            resflex = self.protein.backbone_dynamics[seq_i]

            # average region flexibility
            #stt_i = max(0, seq_i - 9)
            #end_i = min(len(self.protein.protein_sequence), seq_i + 10)
            #avgflex = numpy.mean(self.protein.backbone_dynamics[stt_i: end_i])

            #return [resflex, avgflex]
            return [resflex]

        else:
            ids = ['resflex']
            names = ['Resdue flexibility']
            return(ids, names)

    # feature calculation help functions

    def sequence_environment(self, distance):
        '''
        '''
        cur_index = self.position - 1
        start_index = max(0, cur_index - distance)
        end_index = cur_index + distance + 1
        five_prime = self.protein.protein_sequence[start_index: cur_index]
        three_prime = self.protein.protein_sequence[cur_index + 1: end_index]
        return (five_prime, three_prime)

    def seq_env(self, window, fill_character=None):

        # check for positive uneven window size
        if(window / 2 == 0):
            raise ValueError('window must be uneven.')
        if(window <= 0):
            raise ValueError('window must be positive.')

        # check for single fill_character of type str
        if(fill_character):
            if(not type(fill_character) == str):
                raise ValueError('fill_character must be of type str')
            if(not len(fill_character) == 1):
                raise ValueError('fill_character must be a single character.')

        # mutation index on protein sequence
        cur_index = self.position - 1

        # calculate start and end index for environment subsequence
        distance = window / 2
        start_index = cur_index - distance
        end_index = cur_index + (distance + 1)

        # if window falls of sequence 5-prime side
        prefix = ''
        if(start_index < 0):
            if(fill_character):
                prefix = (-1 * start_index) * fill_character
            start_index = 0

        # if window falls of sequence 3-prime side
        postfix = ''
        diff = end_index - len(self.protein.protein_sequence)
        if(diff > 0):
            if(fill_character):
                postfix = diff * fill_character

        # obtain the subsequence
        subseq = self.protein.protein_sequence[start_index: end_index]

        # return it with appended pre- and postfix
        return prefix + subseq + postfix

    def mutation_signal_distance(self, scale):
        return scale[self.aa_from] - scale[self.aa_to]

    def min_signal_dist_to_msa(self, scale):
        '''
        Returns the minimal distance (for the given scale) of the 'to' amino
        acid to any of the amino acids on the same position in the multiple
        sequence alignment (i.e. the msa variability of this position).
        '''
        var = self.protein.msa_variability(self.position, with_gaps=False)
        distances = [(scale[self.aa_to] - scale[v]) for v in var]
        return min(distances)

    def environment_signal(self, env_window, scale, sig_window, edge):

        # add stub character to scale, to extend loose ends sequence
        fill_character = '#'
        scale[fill_character] = 0.0

        # obtain subsequence (filled at the ends)
        subseq = self.seq_env(env_window, fill_character=fill_character)

        # obtain the signal and append to result
        return sequtil.seq_signal(subseq, scale, sig_window, edge)

    def environment_signal_peak_area(self, env_window, scale, sig_window,
                                     edge, threshold, below_threshold=False):

        # obtain the signal
        signal = self.environment_signal(env_window, scale, sig_window, edge)

        # translate x-axis to threshold y-value
        translated_signal = signal - threshold

        # flip the signal around x-axis if below threshold is requested
        if(below_threshold):
            translated_signal = translated_signal * -1

        # clip negative values to 0
        peaks = translated_signal.clip(0.0)

        # obtain area under the curve using composite trapezoidal rule
        area = scipy.integrate.trapz(peaks)

        return area

    def pfam_family(self):
        return self.protein.pfam_family(self.position)

    def pfam_domain(self):
        return self.protein.pfam_domain(self.position)

    def pfam_repeat(self):
        return self.protein.pfam_repeat(self.position)

    def pfam_clan(self):
        return self.protein.pfam_clan(self.position)

    def pfam_clan_index(self):
        return self.protein.pfam_clan_index(self.position)

    def pfam_active_residue(self):
        return self.protein.pfam_active_residue(self.position)

    # check attribute availability functions (simple getters)

    def get_aa_from(self):
        return self.aa_from

    def get_aa_to(self):
        return self.aa_to

    def get_pdb_id(self):
        return self.pdb_id

    def get_pdb_resnum(self):
        return self.pdb_resnum
