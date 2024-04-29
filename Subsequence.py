from numpy import array
from itertools import combinations, combinations_with_replacement, permutations
import numpy as np


def constructPartitions(seq_list, cpu_num):
    cpu_num = int(cpu_num)
    seqs_num = len(seq_list)
    batch_num = seqs_num//cpu_num
    batches = []
    for i in range(cpu_num-1):
        batch = seq_list[i*batch_num:(i+1)*batch_num]
        batches.append(batch)
    batch=seq_list[(cpu_num-1)*batch_num:]
    batches.append(batch)
    return batches

def getKmerDict(alphabet, k):
    kmerlst = []
    partkmers = list(combinations_with_replacement(alphabet, k))
    for element in partkmers:
        elelst = set(permutations(element, k))
        strlst = [''.join(ele) for ele in elelst]
        kmerlst += strlst
    kmerlst = np.sort(kmerlst)
    kmerdict = {kmerlst[i]:i for i in range(len(kmerlst))}
    return kmerdict


def getSubsequenceProfile(sequence, alphabet, k, delta):
    kmerdict = getKmerDict(alphabet, k)
    vector = getSubsequenceProfileVector(sequence, kmerdict, k, delta)
    vector=array(vector)
    return vector

def getSubsequenceProfileVector(sequence, kmerdict, k, delta):
    vector = np.zeros((1,len(kmerdict)))
    sequence = array(list(sequence))
    n = len(sequence)
    #index_lst = list(combinations(range(n), k))
    for subseq_index in combinations(list(range(n)), k):
        subseq_index = list(subseq_index)
        subsequence = sequence[subseq_index]
        position = kmerdict.get(''.join(subsequence))
        subseq_length = subseq_index[-1] - subseq_index[0] + 1
        subseq_score = 1 if subseq_length == k else delta**subseq_length
        vector[0,position] += subseq_score
    #return list(vector[0])
    return [round(f, 4) for f in list(vector[0])]


# =================getting (k, delta)-subsequence profile=======================
def getSubsequenceProfileByParallel(seq, alphabet, k, delta):
    alphabet = list(alphabet)

    temp = getSubsequenceProfile(seq,alphabet,k,delta)
    return temp



def getSubsequence(seq, alphabet, k, delta):
    Subseqres = getSubsequenceProfileByParallel(seq=seq, alphabet=alphabet, k=k, delta=delta)
    return Subseqres
