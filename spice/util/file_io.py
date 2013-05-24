import os

### not sure if this is realy handy...
from spice import protein

'''
Created on Sep 10, 2011

@author: Bastiaan van den Berg
'''


# This is a generator function
def read_fasta(f, filter_ids=None):
    '''
    '''

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    # HACK to handle response of fasta download from uniprot website
    elif(f.__class__.__name__ == 'addinfourl'):
        handle = f
    else:
        handle = open(f, 'r')

    # initialize sequence id and string to an empty string
    seq_id = ""
    seq_str = ""

    # iterate over each line in the fasta file
    for line in handle:

        if(seq_id == "" and seq_str == ""):
            if(line[0] == ">"):
                seq_id = line.split()[0][1:]
            elif(line[0] == '#'):
                pass
            elif(line.strip()):
                # non-empty line...
                print(line.strip())
                raise(Exception, "Error in fasta file")
        else:
            if(line.strip() == "" or line[0] == ">"):
                if(filter_ids is None or seq_id in filter_ids):
                    yield (seq_id, seq_str)
                seq_str = ""
                if(line[0] == ">"):
                    seq_id = line.split()[0][1:]
                else:
                    seq_id = ""
            else:
                seq_str += line.strip()

    # return the last sequence (not if the file was empty)
    if not(seq_id == ""):
        if(filter_ids is None or seq_id in filter_ids):
            yield (seq_id, seq_str)

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def write_fasta(f, seqs):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'w')

    for s in seqs:
        handle.write('>' + s[0] + '\n')
        for i in range(0, len(s[1]), 70):
            handle.write(s[1][i:i + 70] + '\n')
        handle.write('\n')
        handle.flush()

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def read_pfam(f):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'r')

    # initialize sequence id and list of annotations
    current_id = ""
    current_annotations = []

    # iterate over lines
    for line in handle:

        if(current_id == "" and len(current_annotations) == 0):
            if(line[0] == ">"):
                current_id = line.split()[0][1:]
            elif(line[0] == '#'):
                pass
            elif(line.strip()):
                # non-empty line...
                print(line.strip())
                raise(Exception, "Error in pfam file")
        else:
            if(line.strip() == "" or line[0] == ">"):
                yield (current_id, current_annotations)
                current_annotations = []
                if(line[0] == ">"):
                    current_id = line.split()[0][1:]
                else:
                    current_id = ""
            else:
                annotation = protein.Pfam.parse(line.strip())
                current_annotations.append(annotation)

    # return annotations for the last item (not if the file was empty)
    if not(current_id == ""):
        yield (current_id, current_annotations)

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def write_pfam(f, pfam_data):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'w')

    for uni, pfam in pfam_data:
        handle.write('>%s\n' % (uni))
        for annotation in pfam:
            handle.write('%s\n' % (annotation.single_line_str()))
        handle.write('\n')
        handle.flush()

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def read_mutation(f):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'r')

    types = (str, int, str, str, str, int)

    tuples = []
    for line in handle:
        tokens = line.split()
        row = []
        for index, t in enumerate(types):
            if(index == 4 and tokens[index] == 'None'):
                row.append(None)
            else:
                row.append(t(tokens[index]))
        tuples.append(tuple(row))

    # close file if we opened it
    if not(type(f) == file):
        handle.close()

    return tuples


def write_mutation(f, mutations):
    assert(all([len(m) == 6 for m in mutations]))
    write_tuple_list(f, mutations)


def read_pdb_dir(pdb_fs, pdb_dir):
    '''
    Only .ent files are read...
    Returns AtomGroup object (prody)
    '''

    # import prody for pdb read/write
    import prody
    prody.confProDy(verbosity='none')

    struct_data = []

    for pdb_f in pdb_fs:

        pdb_f = os.path.join(pdb_dir, pdb_f)

        if(os.path.exists(pdb_f)):
            struct = prody.parsePDB(pdb_f)
        else:
            struct = None

        struct_data.append((pdb_f, struct))

    return struct_data


def write_pdb_dir(pdb_dir, struct_data):

    # import prody for pdb read/write
    import prody

    # create output directory, if not yet present
    if not(os.path.exists(pdb_dir)):
        os.makedirs(pdb_dir)

    # write protein chain pdb files to output directory
    for (pdb_f, struct) in struct_data:
        if not(struct is None):
            out_f = os.path.join(pdb_dir, pdb_f)
            prody.writePDB(out_f, struct)


def read_rasa_dir(rasa_fs, rasa_dir):
    rasa_data = []
    for rasa_f in rasa_fs:
        rasa_f = os.path.join(rasa_dir, rasa_f)
        if(os.path.exists(rasa_f)):
            rasa = read_python_list(rasa_f)
        else:
            rasa = None
        rasa_data.append((rasa_f, rasa))
    return rasa_data


def write_rasa_dir(rasa_dir, rasa_data):
    # create output directory, if not yet present
    if not(os.path.exists(rasa_dir)):
        os.makedirs(rasa_dir)

    # write protein chain pdb files to output directory
    for (rasa_f, rasa) in rasa_data:
        if not(rasa is None):
            out_f = os.path.join(rasa_dir, rasa_f)
            write_python_list(out_f, rasa)


def read_python_list(f):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'r')

    # read list (not very neat, but whatever :)
    result = eval(handle.read())
    assert(type(result) == list)
    assert(all([type(item) == float for item in result]))

    if not(type(f) == file):
        handle.close()

    return result


def write_python_list(f, l):

    assert(type(l) == list)
    assert(all([type(item) == float for item in l]))

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'w')
    handle.write('%s\n' % (str(l)))
    if not(type(f) == file):
        handle.close()


def read_residue_rank_dir(rank_fs, rank_dir):

    rank_data = []

    for rank_f in rank_fs:

        rank_f = os.path.join(rank_dir, rank_f)

        if(os.path.exists(rank_f)):
            rank = read_residue_rank(rank_f)
        else:
            rank = None

        rank_data.append((rank_f, rank))

    return rank_data


def write_residue_rank_dir(rank_dir, rank_data):

    if not(os.path.exists(rank_dir)):
        os.makedirs(rank_dir)

    for (rank_f, rank) in rank_data:
        if not(rank is None):
            out_f = os.path.join(rank_dir, rank_f)
            write_residue_rank(out_f, rank)


def read_residue_rank(f):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'r')

    # store results
    result = []

    # iterate over lines in file
    for line in handle:

        # ignore empty lines and comments
        if(line.strip() and not line[0] == '%'):

            # each other line should be composed of 7 items
            tokens = line.split()

            # read the 7 items
            ali_pos = int(tokens[0])
            seq_pos = int(tokens[1])
            aa = tokens[2]
            coverage = float(tokens[3])
            var_count = int(tokens[4])
            var_letters = tokens[5]
            rvet_score = float(tokens[6])

            # store ass tuple and add to result
            result.append((ali_pos, seq_pos, aa, coverage, var_count,
                    var_letters, rvet_score))

    if not(type(f) == file):
        handle.close()

    return result


def write_residue_rank(f, rank_data):
    assert(all([len(r) == 7 for r in rank_data]))
    write_tuple_list(f, rank_data)


def read_classification_result(f, score=None):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'r')

    score_names = handle.readline().split(',')

    score_lists = []
    for sn in score_names:
        score_lists.append(eval(handle.readline()))

    if(score):
        if not(score in score_names):
            raise ValueError('Requested score not available: %s' % (score))
        else:
            result = (score, score_lists[score_names.index(score)])
    else:
        result = (score_names, score_lists)

    if not(type(f) == file):
        handle.close()

    return result


def write_classification_result(f, result):
    # TODO
    pass


def read_labeling(f):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'r')

    class_names = handle.readline().split()

    label_dict = {}
    for line in handle:
        tokens = line.split()
        label = int(tokens[1])
        assert(label < len(class_names))
        label_dict[tokens[0]] = label

    if not(type(f) == file):
        handle.close()

    return (label_dict, class_names)


def write_labeling(f, object_ids, labels, class_names):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'w')

    handle.write('%s\n' % ('\t'.join(class_names)))

    for (obj, lab) in zip(object_ids, labels):
        handle.write('%s\t%s\n' % (obj, lab))

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def read_propka30(filename):

    # values to be read
    feph = []           # 15 times free energy per pH
    chphf = []          # 15 charge per pH folded
    chphu = []          # 15 charge per pH unfolded
    femin = 100000.0
    feminph = -1.0
    pif = -1.0
    piu = -1.0

    # parse status
    first = True
    fe = False
    fe_count = 0
    ch = False
    ch_count = 0
    max_count = 15

    with open(filename, 'r') as fin:

        for line in fin:
            tokens = line.split()
            if(first):
                assert(tokens[0] == 'propka3.0,' and tokens[2] == '182')
                first = False
            if(tokens):
                if(fe and fe_count < max_count):
                    feph.append(float(tokens[1]))
                    fe_count += 1
                if(ch and ch_count < max_count):
                    chphf.append(float(tokens[1]))
                    chphu.append(float(tokens[2]))
                    ch_count += 1
                if(tokens[0] == 'Free'):
                    fe = True
                elif(tokens[0] == 'pH'):
                    ch = True
                if(tokens[0] == 'The' and tokens[1] == 'pI'):
                    pif = float(tokens[3])
                    piu = float(tokens[6])
                if(tokens[0] == 'The' and tokens[1] == 'pH'):
                    feminph = float(tokens[6])
                    femin = float(tokens[13])

    assert(len(feph) == max_count)
    assert(len(chphf) == max_count)
    assert(len(chphu) == max_count)
    assert(not femin == 100000.0)
    assert(not feminph == -1.0)
    assert(not pif == -1.0)
    assert(not piu == -1.0)

    # bit tricky... return as dict???
    return((feph, chphf, chphu, femin, feminph, pif, piu))


# This is a generator function
def read_ids(f):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'r')

    for line in handle:
        tokens = line.split()
        yield(tokens[0])

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def write_ids(f, ids):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'w')

    for uid in ids:
        handle.write('%s\n' % (uid))

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def read_names(f):
    
    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'r')

    for line in handle:
        name = line.strip()
        yield(name)

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def write_names(f, names):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'w')

    for name in names:
        handle.write('%s\n' % (name.strip()))

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def read_dict(handle, value_type, num_cols=1):
    result = {}
    for line in handle:
        tokens = line.split()
        if(num_cols > 1):
            end = num_cols + 2
            result[tokens[0]] = tuple([value_type(i) for i in tokens[1:end]])
        else:
            result[tokens[0]] = value_type(tokens[1])
    return result


def write_dict(handle, d):
    for key in d.keys():
        handle.write('%s\t%s\n' % (key, str(d[key])))


# use eval instead of passing list of types???
def read_tuple_list(f, types):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'r')

    tuples = []
    for line in handle:
        tokens = line.split()
        row = []
        for index in xrange(len(types)):
            row.append(types[index](tokens[index]))
        tuples.append(tuple(row))

    # close file if we opened it
    if not(type(f) == file):
        handle.close()

    return tuples


def write_tuple_list(f, tuple_list):

    # open file if path is provided instead of file
    if(type(f) == file):
        handle = f
    else:
        handle = open(f, 'w')

    for tup in tuple_list:
        for item in tup:
            handle.write('%s\t' % (str(item)))
        handle.write('\n')

    # close file if we opened it
    if not(type(f) == file):
        handle.close()


def read_cross_validation(cv_file):
    '''
    Each line contains a list of test set indices, all other indices are
    assumed to be train set.

    returns list of tuples, each tuple containing a list of train indices
    and a list of test indices, these can be used as cv parameter.
    '''

    # store test indices per CV-fold
    tst_is = []

    # featch the cv-fold test indices from file
    with open(cv_file, 'r') as fin:
        for line in fin:
            tst_is.append([int(t) for t in line.split()])

    # get all indices
    all_is = []
    for i_list in tst_is:
        all_is.extend(i_list)
    all_is_set = set(all_is)

    # test if none of the indices is in multiple sets and make a full range
    assert(len(all_is) == len(all_is_set))
    assert(range(len(all_is)) == sorted(all_is))

    # create cv indices
    cv = []
    for tst_i_list in tst_is:
        trn_i_list = all_is_set - set(tst_i_list)
        cv.append((sorted(trn_i_list), sorted(tst_i_list)))

    return cv


