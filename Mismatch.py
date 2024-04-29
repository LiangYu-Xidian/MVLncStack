from numpy import array
from itertools import combinations_with_replacement, permutations
import numpy as np


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


def getSpectrumProfileVector(sequence, kmerdict, p, k):
    vector = np.zeros((1, p**k))
    n = len(sequence)
    for i in range(n-k+1):
        subsequence=sequence[i:i+k]
        position=kmerdict.get(subsequence)
        vector[0,position] += 1
    return list(vector[0])



def getMismatchProfileVector(sequence, alphabet, kmerdict, p, k):
    n = len(sequence)
    vector = np.zeros((1, p**k))
    for i in range(n-k+1):
        subsequence = sequence[i:i+k]
        position = kmerdict.get(subsequence)
        vector[0, position]+=1
        for j in range(k):
            substitution = subsequence
            for letter in list(set(alphabet)^set(subsequence[j])):
                substitution = list(substitution)
                substitution[j] = letter
                substitution = ''.join(substitution)
                position = kmerdict.get(substitution)
                vector[0,position] += 1
    return list(vector[0])


def getMismatchProfileMatrix(sequence, alphabet, k, m):
    alphabet = list(alphabet)
    p = len(alphabet)

    kmerdict = getKmerDict(alphabet, k)
    features = []
    if m==0 and m < k:
        vector=getSpectrumProfileVector(sequence, kmerdict, p, k)

    elif m > 0 and m < k:
        vector=getMismatchProfileVector(sequence, alphabet, kmerdict, p, k)

    return array(vector)


def getMismatch(seq, alphabet, k, m):
    Mismatchres = getMismatchProfileMatrix(seq, alphabet, k, m)
    return Mismatchres
