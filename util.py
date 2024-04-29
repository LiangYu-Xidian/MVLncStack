#!/usr/bin/env python
# -*- coding: utf-8 -*-


from itertools import count, takewhile
import shutil
from numpy import *
import math
from sklearn.decomposition import PCA
import operator
from imblearn.over_sampling import SVMSMOTE
from functools import reduce


def frange(start, stop, step):
    return takewhile(lambda x: x <= stop, count(start, step))


"""Used for process original data."""


class Seq:
    def __init__(self, name, seq, no):
        self.name = name
        self.seq = seq.upper()
        self.no = no
        self.length = len(seq)

    def __str__(self):
        """Output seq when 'print' method is called."""
        return "%s\tNo:%s\tlength:%s\n%s" % (self.name, str(self.no), str(self.length), self.seq)


class OET_KNN(object):
    """
    @author: Zhangjun
    """

    def __init__(self, n_neighbors=5):
        """
        Initialize the object.
        :param n_neighbors: The number of neighbors to consider, the default value of 5.
        """
        self.n_neighbors = n_neighbors

    def get_CredDis(self, mq_x, mq_y):
        """
        Obtain the confidence distribution of the predicted class for a test sample.
        :param mq_x:
        :param mq_y:
        :return:
        """
        mo_x = 1 - mq_x
        mo_y = 1 - mq_y
        total = mq_x * mq_y + mq_x * mo_y + mq_y * mo_x + mo_x * mo_y
        tmp_q = (mq_x * mq_y + mq_x * mo_y + mq_y * mo_x) / total
        # tmp_o = (mo_x * mo_y) / total
        return tmp_q

    def get_Creds(self, sqDistances, indexes):
        """
        According to the distance to obtain the predictability for this class
        :param sqDistances: Square distance
        :param indexes: Indexes of samples belong to a class in trainSet
        :return:
        """
        creds = []
        for i in indexes:
            d = sqDistances[i]
            mq = exp(-d)
            creds.append(mq)
        return creds

    def nomalizeVector(self, trainset, testset):
        """
        Normalize the feature matrix so that the element values are between 0 and 1
        :param trainset:
        :param testset:
        :return:
        """
        dataset = vstack((trainset, testset))
        # print dataset.shape
        minVals = dataset.min(0)
        maxVals = dataset.max(0)
        ranges = maxVals - minVals
        tmp = []
        for e in ranges:
            if e > 0:
                tmp.append(e)
            else:
                tmp.append(1.0e-10)
        ranges = tmp
        normTrain = zeros(shape(trainset))
        m = trainset.shape[0]
        normTrain = trainset - tile(minVals, (m, 1))
        normTrain = normTrain / tile(ranges, (m, 1))
        normTest = zeros(shape(testset))
        m = testset.shape[0]
        normTest = testset - tile(minVals, (m, 1))
        normTest = normTest / tile(ranges, (m, 1))
        return normTrain, normTest

    def oet_classify(self, trainSet, testSet, labels, is_mult_class=False):
        """
        :param trainSet: TrainSet, Type=arry
        :param testSet: TestSet,Type=arry
        :param labels: Labels of smples in train set.
        :param is_mult_class: If there are multiple class is_mult_class should be True, else it should be False.
        :return:If multiple class, return predicted labels, if two class, return predicted labels and probabilities,
        the default value is False.
        """
        trainSet, testSet = self.nomalizeVector(array(trainSet, dtype=float), array(testSet, dtype=float))
        datasetSize = trainSet.shape[0]
        pred_labels = []
        pred_probability = []
        for inX in testSet:
            diffMat = tile(inX, (datasetSize, 1)) - trainSet
            sqDiffMat = diffMat ** 2
            sqDistances = sqDiffMat.sum(axis=1)
            sortedDistIndicies = argsort(sqDistances)
            classCred = {}
            indexes = []
            for i in range(self.n_neighbors):
                indexes.append(sortedDistIndicies[i])
            creds = self.get_Creds(sqDistances, indexes)
            for i in range(self.n_neighbors):
                voteIlabel = labels[indexes[i]]
                cred = creds[i]
                classCred[voteIlabel] = self.get_CredDis(classCred.get(voteIlabel, 0.0), cred)
            sortedClassCred = sorted(iter(classCred.items()), key=operator.itemgetter(1), reverse=True)
            pred_labels.append(sortedClassCred[0][0])
            s = 0.0
            for e in sortedClassCred:
                s += float(e[1])
            pred_probability.append(float(sortedClassCred[0][1]) / s)
        if not is_mult_class:
            tmp_prob = []
            for i in range(len(pred_labels)):
                if pred_labels[i] > 0:
                    tmp_prob.append(pred_probability[i])
                else:
                    tmp_prob.append(1 - pred_probability[i])
            pred_probability = tmp_prob
            return pred_labels, pred_probability
        else:
            return pred_labels


# ---------------------------------------------------------------------
# Covariance Discriminant algorithm
# ---------------------------------------------------------------------

def split_trainSet(trainSet, labels):
    trianDic = {}
    for i in range(len(labels)):
        lb = labels[i]
        tmp = trianDic.get(lb, 0)
        if tmp == 0:
            trianDic[lb] = [i]
        else:
            trianDic[lb].append(i)
    set_list = []
    label_list = []
    for clas in list(trianDic.items()):
        label_list.append(clas[0])
        tmpMat = []
        # print clas[1]
        for i in clas[1]:
            tmpMat.append(trainSet[i])
        set_list.append(np.array(tmpMat))
    return set_list, label_list


def nomalizeVector(trainset, testset):
    dataset = vstack((trainset, testset))
    U, Simga, VT = np.linalg.svd(dataset)
    N = 0
    tmp = 0.0
    for e in Simga:
        tmp += e
        N += 1
        if tmp / sum(Simga) > 0.99:
            break
    pca = PCA(n_components=N)
    sim_mat = pca.fit(dataset).transform(dataset)
    # sim_mat=dataset
    trainset = sim_mat[:trainset.shape[0]]
    testset = sim_mat[trainset.shape[0]:]
    minVals = sim_mat.min(0)
    maxVals = sim_mat.max(0)
    ranges = maxVals - minVals
    normTrain = zeros(shape(trainset))
    m = trainset.shape[0]
    normTrain = trainset - tile(minVals, (m, 1))
    normTrain = normTrain / tile(ranges, (m, 1))
    normTest = zeros(shape(testset))
    m = testset.shape[0]
    normTest = testset - tile(minVals, (m, 1))
    normTest = normTest / tile(ranges, (m, 1))
    return normTrain, normTest


def get_covariance_matrix(mat):
    aver_vector = np.average(mat, axis=0)
    covarMat = np.zeros([mat.shape[1], mat.shape[1]], dtype=float)
    for i in range(mat.shape[1]):
        for j in range(mat.shape[1]):
            s = 0.0
            for k in range(mat.shape[0]):
                s += (mat[k][i] - aver_vector[i]) * (mat[k][j] - aver_vector[j])
            covarMat[i][j] = s / (mat.shape[0] - 1)
    return covarMat


def cda(trainSet, labels, testSet, is_mult_class=False):
    trainSet = np.array(trainSet)
    testSet = np.array(testSet)
    trainSet, testSet = nomalizeVector(array(trainSet), array(testSet))
    trainSet_list, class_list = split_trainSet(trainSet, labels)
    covarMat_list = []
    averVect_list = []
    for i in range(len(class_list)):
        covarMat = get_covariance_matrix(trainSet_list[i])
        aver_vector = np.average(trainSet_list[i], axis=0)
        covarMat_list.append(covarMat)
        averVect_list.append(aver_vector)
    pred_labels = []
    pred_probs = []
    for query in testSet:
        pred_label = []
        for i in range(len(class_list)):
            covarMat = covarMat_list[i]
            aver_vector = averVect_list[i]
            asb_a = np.linalg.eigvals(covarMat)
            list(asb_a).sort(reverse=True)
            N = 0
            tmp = 0.0
            for e in asb_a:
                tmp += e
                N += 1
                if tmp / sum(asb_a) > 0.9:
                    break
            asb_a = asb_a[:N]
            C = reduce((lambda x, y: x * y), asb_a)
            if C <= 0:
                C = 1
            cred = np.transpose(query - aver_vector).dot(np.linalg.inv(covarMat)).dot(query - aver_vector) \
                   + math.log(C) - 2 * math.log(float(trainSet_list[i].shape[0]) / float(trainSet.shape[0]))
            pred_label.append(cred)
        prob = (sum(pred_label) - np.min(pred_label)) / sum(pred_label)
        pred_probs.append(prob)
        pred_labels.append(class_list[pred_label.index(np.min(pred_label))])
    if not is_mult_class:
        tmp_prob = []
        for i in range(len(pred_labels)):
            if pred_labels[i] > 0:
                tmp_prob.append(pred_probs[i])
            else:
                tmp_prob.append(1 - pred_probs[i])
        return pred_labels, tmp_prob
    else:
        return pred_labels


# ---------------------------------------------------------------------
# Covariance Discriminant algorithm ends.
# ---------------------------------------------------------------------

def is_under_alphabet(s, alphabet):
    """Judge the string is within the scope of the alphabet or not.

    :param s: The string.
    :param alphabet: alphabet.

    Return True or the error character.
    """
    for e in s:
        if e not in alphabet:
            return e

    return True


def is_fasta(seq):
    """Judge the Seq object is in FASTA format.
    Two situation:
    1. No seq name.
    2. Seq name is illegal.
    3. No sequence.

    :param seq: Seq object.
    """
    if not seq.name:
        error_info = 'Error, sequence ' + str(seq.no) + ' has no sequence name.'
        print(seq)
        sys.stderr.write(error_info)
        return False
    if -1 != seq.name.find('>'):
        error_info = 'Error, sequence ' + str(seq.no) + ' name has > character.'
        sys.stderr.write(error_info)
        return False
    if 0 == seq.length:
        error_info = 'Error, sequence ' + str(seq.no) + ' is null.'
        sys.stderr.write(error_info)
        return False

    return True


def read_fasta(f):
    """Read a fasta file.

    :param f: HANDLE to input. e.g. sys.stdin, or open(<file>)

    Return Seq obj list.
    """
    name, seq = '', ''
    count = 0
    seq_list = []
    lines = f.readlines()
    for line in lines:
        if not line:
            break

        if '>' == line[0]:
            if 0 != count or (0 == count and seq != ''):
                if is_fasta(Seq(name, seq, count)):
                    seq_list.append(Seq(name, seq, count))
                else:
                    sys.exit(0)

            seq = ''
            name = line[1:].strip()
            count += 1
        else:
            seq += line.strip()

    count += 1
    if is_fasta(Seq(name, seq, count)):
        seq_list.append(Seq(name, seq, count))
    else:
        sys.exit(0)

    return seq_list


def read_fasta_yield(f):
    """Yields a Seq object.

    :param f: HANDLE to input. e.g. sys.stdin, or open(<file>)
    """
    name, seq = '', ''
    count = 0
    while True:
        line = f.readline()
        if not line:
            break

        if '>' == line[0]:
            if 0 != count or (0 == count and seq != ''):
                if is_fasta(Seq(name, seq, count)):
                    yield Seq(name, seq, count)
                else:
                    sys.exit(0)

            seq = ''
            name = line[1:].strip()
            count += 1
        else:
            seq += line.strip()

    if is_fasta(Seq(name, seq, count)):
        yield Seq(name, seq, count)
    else:
        sys.exit(0)


def read_fasta_check_dna(f, alphabet):
    """Read the fasta file, and check its legality.

    :param f: HANDLE to input. e.g. sys.stdin, or open(<file>)

    Return the seq list.
    """
    seq_list = []
    for e in read_fasta_yield(f):
        res = is_under_alphabet(e.seq, alphabet)
        if res:
            seq_list.append(e)
        else:
            error_info = 'Sorry, sequence ' + str(e.no) \
                         + ' has character ' + str(res) + '.(The character must be ' + alphabet + ').'
            sys.exit(error_info)

    return seq_list


def get_sequence_check_dna(f, alphabet):
    """Read the fasta file.

    Input: f: HANDLE to input. e.g. sys.stdin, or open(<file>)

    Return the sequence list.
    """
    sequence_list = []
    for e in read_fasta_yield(f):
        res = is_under_alphabet(e.seq, alphabet)
        if res is not True:
            error_info = 'Error, sequence ' + str(e.no) \
                         + ' has character ' + str(res) + '.(The character must be ' + alphabet + ').'
            sys.exit(error_info)
        else:
            sequence_list.append(e.seq)

    return sequence_list


def is_sequence_list(sequence_list, alphabet):
    """Judge the sequence list is within the scope of alphabet and change the lowercase to capital."""
    count = 0
    new_sequence_list = []

    for e in sequence_list:
        e = e.upper()
        count += 1
        res = is_under_alphabet(e, alphabet)
        if res is not True:
            error_info = 'Sorry, sequence ' + str(count) \
                         + ' has illegal character ' + str(res) + '.(The character must be A, C, G or T)'
            sys.stderr.write(error_info)
            return False
        else:
            new_sequence_list.append(e)

    return new_sequence_list


def get_data(input_data, alphabet, desc=False):
    """Get sequence data from file or list with check.

    :param input_data: type file or list
    :param desc: with this option, the return value will be a Seq object list(it only works in file object).
    :return: sequence data or shutdown.
    """
    if hasattr(input_data, 'read'):
        if desc is False:
            return get_sequence_check_dna(input_data, alphabet)
        else:
            return read_fasta_check_dna(input_data, alphabet)
    elif isinstance(input_data, list):
        input_data = is_sequence_list(input_data, alphabet)
        if input_data is not False:
            return input_data
        else:
            sys.exit(0)
    else:
        error_info = 'Sorry, the parameter in get_data method must be list or file type.'
        sys.exit(error_info)


"""Some basic function for generate feature vector."""


def frequency(tol_str, tar_str):
    """Generate the frequency of tar_str in tol_str.

    :param tol_str: mother string.
    :param tar_str: substring.
    """
    i, j, tar_count = 0, 0, 0
    len_tol_str = len(tol_str)
    len_tar_str = len(tar_str)
    while i < len_tol_str and j < len_tar_str:
        if tol_str[i] == tar_str[j]:
            i += 1
            j += 1
            if j >= len_tar_str:
                tar_count += 1
                i = i - j + 1
                j = 0
        else:
            i = i - j + 1
            j = 0

    return tar_count


def frequency_p(tol_str, tar_str):
    """Generate the frequency of tar_str in tol_str.

    :param tol_str: mother string.
    :param tar_str: substring.
    """
    i, j, tar_count, tar1_count, tar2_count, tar3_count = 0, 0, 0, 0, 0, 0
    tar_list = []
    len_tol_str = len(tol_str)
    len_tar_str = len(tar_str)
    while i < len_tol_str and j < len_tar_str:
        if tol_str[i] == tar_str[j]:
            i += 1
            j += 1
            if j >= len_tar_str:
                tar_count += 1
                i = i - j + 1
                j = 0
                if (
                        i + 1) % 3 == 1:  # judge the position of last base of kmer in corresponding codon. pay attention to "i + 1"

                    tar1_count += 1
                elif (i + 1) % 3 == 2:
                    tar2_count += 1
                else:
                    tar3_count += 1
        else:
            i = i - j + 1
            j = 0
    tar_list = (tar_count, tar1_count, tar2_count, tar3_count)
    return tar_list


def Z_curve(sequence, k, alphabet):
    kmer = make_kmer_list(k, alphabet)
    len_kmer = len(kmer)
    i = 0
    f_ZC = []
    fx_list = []
    fy_list = []
    fz_list = []
    while i < len_kmer:
        j = 1
        fre1_list = []
        fre2_list = []
        fre3_list = []
        while j <= 4:
            fre1 = frequency_p(sequence, str(kmer[i]))[1]
            fre2 = frequency_p(sequence, str(kmer[i]))[2]
            fre3 = frequency_p(sequence, str(kmer[i]))[3]
            fre1_list.append(fre1)
            fre2_list.append(fre2)
            fre3_list.append(fre3)
            j += 1
            i += 1
        fx1 = (fre1_list[0] + fre1_list[2]) - (fre1_list[1] + fre1_list[3])
        fx2 = (fre2_list[0] + fre2_list[2]) - (fre2_list[1] + fre2_list[3])
        fx3 = (fre3_list[0] + fre3_list[2]) - (fre3_list[1] + fre3_list[3])
        fx_list.append(fx1)
        fx_list.append(fx2)
        fx_list.append(fx3)
        fy1 = (fre1_list[0] + fre1_list[1]) - (fre1_list[2] + fre1_list[3])
        fy2 = (fre2_list[0] + fre2_list[1]) - (fre2_list[2] + fre2_list[3])
        fy3 = (fre3_list[0] + fre3_list[1]) - (fre3_list[2] + fre3_list[3])
        fy_list.append(fy1)
        fy_list.append(fy2)
        fy_list.append(fy3)
        fz1 = (fre1_list[0] + fre1_list[3]) - (fre1_list[1] + fre1_list[2])
        fz2 = (fre2_list[0] + fre2_list[3]) - (fre2_list[1] + fre2_list[2])
        fz3 = (fre3_list[0] + fre3_list[3]) - (fre3_list[1] + fre3_list[2])
        fz_list.append(fz1)
        fz_list.append(fz2)
        fz_list.append(fz3)
    for i in range(0, len(fx_list)):
        f_ZC.append(fx_list[i])
    for i in range(0, len(fy_list)):
        f_ZC.append(fy_list[i])
    for i in range(0, len(fz_list)):
        f_ZC.append(fz_list[i])

    return f_ZC


def write_libsvm(vector_list, label_list, write_file):
    """Write the vectors into disk in livSVM format."""
    len_vector_list = len(vector_list)
    len_label_list = len(label_list)
    # print(label_list)
    if len_vector_list == 0:
        sys.exit("The vector is none.")
    if len_label_list == 0:
        sys.exit("The label is none.")
    if len_vector_list != len_label_list:
        sys.exit("The length of vector and label is different.")

    with open(write_file, 'w') as f:
        for ind1, vec in enumerate(vector_list):
            temp_write = str(label_list[ind1])
            for ind2, val in enumerate(vec):
                temp_write += ' ' + str(ind2 + 1) + ':' + str(vec[ind2])
            f.write(temp_write)
            f.write('\n')


def write_tab(_vecs, write_file):
    """Write the vectors into disk in tab format."""
    with open(write_file, 'w') as f:
        for vec in _vecs:
            f.write(str(vec[0]))
            for val in vec[1:]:
                f.write('\t' + str(val))
            f.write('\n')


def write_csv(_vecs, write_file):
    """Write the vectors into disk in csv format."""
    import csv
    print(write_file)
    with open(write_file, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for vec in _vecs:
            spamwriter.writerow(vec)


def write_to_file(vectors, out_format, label, outputfile):
    if out_format == 'svm':
        write_libsvm(vectors, [label] * len(vectors), outputfile)
    elif out_format == 'tab':
        write_tab(vectors, outputfile)
    elif out_format == 'csv':
        write_csv(vectors, outputfile)


def convert_phyche_index_to_dict(phyche_index, alphabet):
    """Convert phyche index from list to dict."""
    # for e in phyche_index:
    #     print e
    len_index_value = len(phyche_index[0])
    k = 0
    for i in range(1, 10):
        if len_index_value < 4 ** i:
            sys.exit("Sorry, the number of each index value is must be 4^k.")
        if len_index_value == 4 ** i:
            k = i
            break
    kmer_list = make_kmer_list(k, alphabet)
    # print kmer_list
    len_kmer = len(kmer_list)
    phyche_index_dict = {}
    for kmer in kmer_list:
        phyche_index_dict[kmer] = []
    # print phyche_index_dict
    phyche_index = list(zip(*phyche_index))
    for i in range(len_kmer):
        phyche_index_dict[kmer_list[i]] = list(phyche_index[i])

    return phyche_index_dict


def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError


def standard_deviation(value_list):
    """Return standard deviation."""
    from math import sqrt
    from math import pow
    n = len(value_list)
    average_value = sum(value_list) * 1.0 / n
    return sqrt(sum([pow(e - average_value, 2) for e in value_list]) * 1.0 / (n - 1))


def normalize_index(phyche_index, alphabet, is_convert_dict=False):
    """Normalize the physicochemical index."""
    normalize_phyche_value = []
    for phyche_value in phyche_index:
        average_phyche_value = sum(phyche_value) * 1.0 / len(phyche_value)
        sd_phyche = standard_deviation(phyche_value)
        normalize_phyche_value.append([round((e - average_phyche_value) / sd_phyche, 2) for e in phyche_value])

    if is_convert_dict is True:
        return convert_phyche_index_to_dict(normalize_phyche_value, alphabet)

    print(normalize_phyche_value)
    return normalize_phyche_value


def read_k(alphabet, _method, k):
    import const
    if alphabet == 'Protein':
        return 1
    elif alphabet == 'RNA':
        return 2

    if _method in const.K_2_DNA_METHODS:
        return 2
    elif _method in const.K_3_DNA_METHODS:
        return 3
    elif _method == 'PseKNC' or _method == 'ZCPseKNC' or _method == 'Gene2vec':
        return k
    else:
        print("Error in read_k.")


def check_args(args, filename):
    """Check pse and acc method args."""
    import const
    if 'w' in args:
        if args.w < 0 or args.w > 1:
            print("Error: The value of w must be no less than 0 and no larger than 1.")
            return False
    if 'method' in args:
        if args.alphabet == 'DNA' and args.method not in const.METHODS_DNA:
            if filename == const.ACC_FILENAME:
                print(("Error: the DNA method parameter can only be " + str(const.METHODS_DNA_ACC)))
            if filename == const.PSE_FILENAME:
                print(("Error: the DNA method parameter can only be " + str(const.METHODS_DNA_PSE)))
            else:
                print("Error: the DNA method parameter error.")
            return False
        elif args.alphabet == 'RNA' and args.method not in const.METHODS_RNA:
            if filename == const.ACC_FILENAME:
                print(("Error: the RNA method parameter can only be " + str(const.METHODS_RNA_ACC)))
            if filename == const.PSE_FILENAME:
                print(("Error: the RNA method parameter can only be " + str(const.METHODS_RNA_PSE)))
            else:
                print("Error: the RNA method parameter error.")
            return False
        elif args.alphabet == 'Protein' and args.method not in const.METHODS_PROTEIN:
            if filename == const.ACC_FILENAME:
                print(("Error: the protein method parameter can only be " + str(const.METHODS_PROTEIN_ACC)))
            if filename == const.PSE_FILENAME:
                print(("Error: the protein method parameter can only be " + str(const.METHODS_PROTEIN_PSE)))
            else:
                print("Error: the protein method parameter error.")
            return False
    if 'k' in args:
        if args.k <= 0:
            print("Error: the value of k must be an inter and larger than 0.")
            return False
    return True


# For Pse-in-One-Analysis.
def check_contain_chinese(check_str):
    """Check if the path name and file name user input contain Chinese character.
    :param check_str: string to be checked.
    """
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def del_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)


def copy_file(src_file, dest_file):
    if os.path.isfile(src_file):
        shutil.copyfile(src_file, dest_file)


# For BioSeq-Analysis: sampling, bootstrapping,etc start
def tab2space(input_files):
    out_files = []
    for input_file in input_files:
        file_elem_list = list(os.path.splitext(input_file))
        input_file = file_elem_list[0] + '_tab' + file_elem_list[1]
        out_file = file_elem_list[0] + '_space' + file_elem_list[1]
        with open(input_file) as f, open(out_file, 'w') as w:
            tab_lines = f.readlines()
            for i in range(len(tab_lines)):
                if tab_lines[i][0] == '>':
                    w.writelines(' '.join(tab_lines[i + 1].strip().split('\t')) + '\n')
        out_files.append(out_file)
    return out_files


def list2libsvm(lists):
    libsvm_list = []
    for l in lists:
        i = iter(list(range(1, len(l) + 1)))
        j = iter(l)
        d = dict(zip(i, j))
        libsvm_list.append(d)
    return libsvm_list


def libsvm2list(lists):
    new_lists = []
    for l in lists:
        new_lists.append(list(l.values()))
    return new_lists


def pre_process_data(lines):
    seq = ''
    pro_lines = []
    for line in lines:
        if line[0] == '>':
            if seq != '':
                pro_lines.append(seq + '\n')
                seq = ''
            pro_lines.append(line)
        elif line != lines[-1]:
            seq += line.strip()
        else:
            seq += line
            pro_lines.append(seq)
    return pro_lines


def undersampling(input_files):
    entries = []
    undersampling_files = []
    for input_file in input_files:
        i = 0
        with open(input_file) as f:
            lines = f.readlines()
            lines = pre_process_data(lines)
            for line in lines:
                if line[0] == '>':
                    i += 1
        entries.append(i)
    sample_size = min(entries)
    for i in range(len(input_files)):
        input_file_list = os.path.split(input_files[i])
        undersampling_file = input_file_list[0] + '/' + 'us_' + input_file_list[1]
        undersampling_files.append(undersampling_file)
        with open(input_files[i]) as f, open(undersampling_file, 'w') as w:
            input_file = f.readlines()
            input_file = pre_process_data(input_file)
            if entries[i] == sample_size:
                w.writelines(input_file)
                continue
            random_indices = np.random.choice(entries[i], sample_size, replace=False)
            random_indices = random_indices.tolist()
            print(random_indices)
            num = 0
            flag = False
            for line in input_file:
                if line[0] == '>':
                    if num in random_indices:
                        flag = True
                        w.writelines(line)
                    else:
                        flag = False
                    num += 1
                elif flag:
                    w.writelines(line)

    return undersampling_files


#
# def oversampling(vectors, labels):
#     X = []
#     y = []
#     for vector, label in zip(vectors, labels):
#         X.extend(array(vector))
#         y.extend([label for _ in vector])
#     X = array(X)
#     y = array(y)
#     # Apply regular SMOTE
#     # kind = ['regular', 'borderline1', 'borderline2', 'svm']
#     X_resampled, y_resampled = SMOTE(kind='regular').fit_sample(X, y)
#     X_resampled = X_resampled.tolist()
#     y_resampled = y_resampled.tolist()
#     X_ = []
#     for label in labels:
#         x_label = [x for x, l in zip(X_resampled, y_resampled) if l == label]
#         X_.append(x_label)
#     return X_, y_resampled

def oversampling(vectors, labels):
    X = []
    y = []
    for vector, label in zip(vectors, labels):
        X.append(array(vector))
        y.append(label)
    X = array(X)
    y = array(y)


    # Apply regular SMOTE
    # kind = ['regular', 'borderline1', 'borderline2', 'svm']
    # X_resampled, y_resampled = SMOTE(kind='regular','borderline-2').fit_sample(X, y)
    # X_resampled, y_resampled = SMOTE(random_state=42, k_neighbors=10, m_neighbors=20, kind='svm').fit_sample(X, y)
    X_resampled, y_resampled = SVMSMOTE(random_state=42).fit_resample(X, y)
    X_resampled = X_resampled.tolist()
    y_resampled = y_resampled.tolist()
    return X_resampled, y_resampled


def write_arff_csv(write_file):
    write_file = write_file.split('.')[:-1]
    arff_file = '.'.join(write_file) + '.arff'
    csv_file = '.'.join(write_file) + '.csv'
    with open(arff_file, 'w') as w, open(csv_file, 'w') as w1, open('.'.join(write_file) + '_tab.txt') as f:
        tab_lines = f.readlines()
        float_lines = []
        for i in range(len(tab_lines)):
            if tab_lines[i][0] == '>':
                float_lines.append([x for x in tab_lines[i + 1].strip().split('\t')])
        length = len(float_lines[0])
        w.writelines('@relation vector\n')
        for l in range(length):
            w.writelines('@attribute ' + str(l) + ' numeric\n')
        w.writelines('@data\n')
        for line in float_lines:
            w.writelines(','.join(line) + '\n')
            w1.writelines(','.join(line) + '\n')

    temp_index = arff_file.split('/').index('temp')
    temp_index1 = csv_file.split('/').index('temp')
    arff_file = '/'.join(arff_file.split('/')[temp_index:])
    csv_file = '/'.join(csv_file.split('/')[temp_index1:])
    return arff_file, csv_file


def ss_window_truncated(input_files):
    list_len = []
    ss_files = []

    for input_file in input_files:
        with open(input_file) as f:
            lines = f.readlines()
            lines = pre_process_data(lines)
        for e in lines:
            if e[0] != '>':
                list_len.append(len(e.strip()))
    min_len = min(list_len)
    for input_file in input_files:
        input_file_list = os.path.split(input_file)
        wt_file = input_file_list[0] + '/' + 'wt_' + input_file_list[1]
        ss_files.append(wt_file)
        with open(wt_file, 'w') as w, open(input_file) as f:
            lines = f.readlines()
            lines = pre_process_data(lines)
            for line in lines:
                if line[0] == '>':
                    w.writelines(line)
                else:
                    w.writelines(line.strip()[:min_len] + '\n')

    return ss_files


def if_fold_to_bootstrap(input_files):
    count = 0
    for input_file in input_files:
        with open(input_file) as f:
            lines = f.readlines()
            lines = pre_process_data(lines)
            count += len(lines)
    if count < 200:
        return True
    return False


# For BioSeq-Analysis: sampling, bootstrapping,etc end
