import const
import util
import index_list
import pse
import re

def make_ac_vec(sequence, lag, phyche_value, k):

    # Get the length of phyche_vals.
    phyche_values = list(phyche_value.values())
    len_phyche_value = len(phyche_values[0])

    len_seq = len(sequence)
    each_vec = []

    for temp_lag in range(1, lag + 1):
        for j in range(len_phyche_value):

            # Calculate average phyche_value for a nucleotide.
            ave_phyche_value = 0.0
            for i in range(len_seq - k):
                nucleotide = sequence[i: i + k]
                ave_phyche_value += float(phyche_value[nucleotide][j])
            ave_phyche_value /= (len_seq - k)

            # Calculate the vector.
            temp_sum = 0.0
            for i in range(len_seq - temp_lag - k + 1):
                nucleotide1 = sequence[i: i + k]
                nucleotide2 = sequence[i + temp_lag: i + temp_lag + k]
                temp_sum += (float(phyche_value[nucleotide1][j]) - ave_phyche_value) * (
                    float(phyche_value[nucleotide2][j]))

            each_vec.append(round(temp_sum / (len_seq - temp_lag - k + 1), 8))

    return each_vec

def make_cc_vec(sequence, lag, phyche_value, k):
    phyche_values = list(phyche_value.values())
    len_phyche_value = len(phyche_values[0])

    len_seq = len(sequence)
    each_vec = []


    for temp_lag in range(1, lag + 1):
        for i1 in range(len_phyche_value):
            for i2 in range(len_phyche_value):
                if i1 != i2:
                    # Calculate average phyche_value for a nucleotide.
                    ave_phyche_value1 = 0.0
                    ave_phyche_value2 = 0.0
                    for j in range(len_seq - k):
                        nucleotide = sequence[j: j + k]
                        ave_phyche_value1 += float(phyche_value[nucleotide][i1])
                        ave_phyche_value2 += float(phyche_value[nucleotide][i2])
                    ave_phyche_value1 /= (len_seq - k)
                    ave_phyche_value2 /= (len_seq - k)

                    # Calculate the vector.
                    temp_sum = 0.0
                    for j in range(len_seq - temp_lag - k + 1):
                        nucleotide1 = sequence[j: j + k]
                        nucleotide2 = sequence[j + temp_lag: j + temp_lag + k]
                        temp_sum += (float(phyche_value[nucleotide1][i1]) - ave_phyche_value1) * \
                                    (float(phyche_value[nucleotide2][i2]) - ave_phyche_value2)
                    each_vec.append(round(temp_sum / (len_seq - temp_lag - k + 1), 8))

    return each_vec

def make_acc_vec(seq, lag, phyche_values, k):
    return make_ac_vec(seq, lag, phyche_values, k) + make_cc_vec(seq, lag, phyche_values, k)


def getValues(prop, supInfo):
    values = ""
    name = re.search(prop, supInfo)
    if name:
        strr = prop + '\s*\,(.+)'
        b = re.search(strr, supInfo)
        if b:
            values = b.group(1)
    return values

def getSpecificValue(olinuc, olinucs, prop, supInfo):
    # olinucs = getNucSeq(SupFileName).split(",")
    olinucs = olinucs.strip().split(",")
    values = getValues(prop, supInfo).rstrip()
    values = values.strip().split(",")
    # valueS = [float(x) for x in values.split(",")]
    count = olinucs.index(olinuc)
    value = values[count]
    return float(value)


def avgP(seq, olinucs, length, k, prop, supInfo):
    limit = length - k + 1
    i = 1
    sum = 0
    while i < limit or i == limit:
        # value = hn(seq[i - 1],prop,SupFileName)
        value = getSpecificValue(seq[i - 1], olinucs, prop, supInfo)
        sum = sum + value
        i = i + 1
    sum = sum / limit
    return sum

# autocorrelation: moreau,geary,moran
def moran(seq, olinucs, length, k, l, prop, supInfo):
    limit = length - k - l + 1
    j = 1
    top = 0
    avg = avgP(seq, olinucs, length, k, prop, supInfo)
    while j < limit or j == limit:
        current = getSpecificValue(seq[j - 1], olinucs, prop, supInfo)
        # hn(seq[j-1],prop,SupFileName)
        partOne = current - avg
        next = getSpecificValue(seq[j + l - 1], olinucs, prop, supInfo)
        # hn(seq[j+l-1],prop,SupFileName)
        partTwo = next - avg
        top = top + (partOne * partTwo)
        j = j + 1
    top = top / limit
    limit2 = length - k + 1
    bottom = 0
    b = 1
    while b < limit2 or b == limit2:
        current = getSpecificValue(seq[b - 1], olinucs, prop, supInfo)
        # hn(seq[b-1],prop,SupFileName)
        bottom = bottom + ((current - avg) * (current - avg))
        b = b + 1
    bottom = bottom / limit2
    final = top / bottom
    return final

def geary(seq, olinucs, length, k, l, prop, supInfo):
    lim = length - k + 1
    limit = length - k - l + 1
    b = 1
    sqr = 0
    while b < limit or b == limit:
        current = getSpecificValue(seq[b - 1], olinucs, prop, supInfo)
        # hn(seq[b-1],prop,SupFileName)
        next = getSpecificValue(seq[b + l - 1], olinucs, prop, supInfo)
        # hn(seq[b+l-1],prop,SupFileName)
        sqr = sqr + ((current - next) * (current - next))
        b = b + 1
    top = sqr * lim
    limit2 = (length - k - l + 1)
    c = 1
    sqr2 = 0
    while c < limit2 or c == limit2:
        # current = hn(seq[c-1],prop,SupFileName)
        current = getSpecificValue(seq[c - 1], olinucs, prop, supInfo)
        avg = avgP(seq, olinucs, length, k, prop, supInfo)
        sqr2 = sqr2 + (current - avg) * (current - avg)
        c = c + 1
    bottom = sqr2 * limit * 2
    final = float((top / bottom) * 1000) / 1000.0
    return final

def moreau(seq, olinucs, length, k, l, prop, supInfo):
    limit = length - k - l + 1
    d = 1
    prod = 0
    while d < limit or d == limit:
        current = getSpecificValue(seq[d - 1], olinucs, prop, supInfo)
        # hn(seq[d-1],prop,SupFileName)
        next = getSpecificValue(seq[d + l - 1], olinucs, prop, supInfo)
        # hn(seq[d+l-1],prop,SupFileName)
        prod = prod + (current * next)
        d = d + 1
    final = prod / limit
    return final



def acc(seq, k, lag, phyche_list, alphabet, extra_index_file=None, all_prop=False, theta_type=1):
    """This is a complete acc in PseKNC.

    :param k: int, the value of k-tuple.
    :param phyche_list: list, the input physicochemical properties list.
    :param extra_index_file: a file path includes the user-defined phyche_index.
    :param all_prop: bool, choose all physicochemical properties or not.
    :param theta_type: the value 1, 2 and 3 for ac, cc or acc.
    """
    phyche_list = pse.get_phyche_list(k, phyche_list,
                                      extra_index_file=extra_index_file, alphabet=alphabet, all_prop=all_prop)
    # print(phyche_list)
    # Get phyche_vals.
    if alphabet == index_list.DNA or alphabet == index_list.RNA:

        if extra_index_file is not None:
            extra_phyche_index = pse.get_extra_index(extra_index_file)
            from .util import normalize_index
            phyche_vals = pse.get_phyche_value(k, phyche_list, alphabet,
                                               normalize_index(extra_phyche_index, alphabet, is_convert_dict=True))
        else:
            phyche_vals = pse.get_phyche_value(k, phyche_list, alphabet)


    #DAC是theta_type == 1
    if theta_type == 1:
        return make_ac_vec(seq, lag, phyche_vals, k)
    elif theta_type == 2:
        return make_cc_vec(seq, lag, phyche_vals, k)
    elif theta_type == 3:
        return make_acc_vec(seq, lag, phyche_vals, k)

def sepSequence(seq, k):
    i = k - 1
    seqq = []
    while i < len(seq):
        j = 0
        nuc = ''
        while j < k:
            nuc = seq[i - j] + nuc
            j = j + 1
        seqq.append(nuc)
        i += 1
    return seqq


def autocorrelation(autoc, sequence, props, k, l, alphabet):
    if not props:
        error_info = 'Error, The phyche_list, extra_index_file and all_prop can\'t be all False.'
        raise ValueError(error_info)

    # Getting supporting info from files
    if k == 2 and alphabet == index_list.RNA:
        SupFileName = './data/Supporting_Information_S1_RNA.txt'

    SupFile = open(SupFileName, 'r')
    supInfo = SupFile.read()
    o = re.search('Physicochemical properties\,(.+)\n', supInfo)
    olinucs = ''
    if o:
        olinucs = o.group(1).rstrip()
    SupFile.close()

    length = len(sequence)
    seq = sepSequence(sequence, k)
    values = []
    for prop in props:
        if autoc.upper() == 'MAC':
            value = float("%.3f" % moran(seq, olinucs, length, k, l, prop, supInfo))
            values.append(value)
        elif autoc.upper() == 'GAC':
            value = float("%.3f" % geary(seq, olinucs, length, k, l, prop, supInfo))
            values.append(value)
        elif autoc.upper() == 'NMBAC':
            value = float("%.3f" % moreau(seq, olinucs, length, k, l, prop, supInfo))
            values.append(value)

    return values




def make_DAC_vector(seq,  method, lag, lamada):
    alphabet = 'RNA'

    if method.upper() not in ['MAC', 'GAC', 'NMBAC', 'PDT']:

        k = util.read_k(alphabet, method, 0)

        # Set Pse default index_list.
        if alphabet == 'RNA':
            alphabet_list = index_list.RNA
            default_e = const.DI_INDS_RNA


        if method in const.METHODS_AC:
            theta_type = 1
        elif method in const.METHODS_CC:
            theta_type = 2
        elif method in const.METHODS_ACC:
            theta_type = 3
        else:
            print("Method error!")
            return False

        # ACC.
        res = acc(seq, k, lag, default_e, alphabet_list, extra_index_file=None, all_prop=False, theta_type=theta_type)
        return res

    elif method.upper() in ['MAC',  'NMBAC']:

        if alphabet == 'RNA':
            alphabet = index_list.RNA
            res = autocorrelation(autoc=method, sequence=seq, props=const.DEFAULT_RNA_IND, k=2, l=lamada, alphabet=alphabet)
            return res
