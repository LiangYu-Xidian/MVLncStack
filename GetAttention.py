import math

import torch.nn as nn

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional
import time
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# s(x, q) = v.T * tanh (W * x + b)
class ML_AttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, attention_dim, src_length_masking=True):
        super(ML_AttentionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.src_length_masking = src_length_masking

        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, x, x_lengths):
        """
        :param x: seq_len * batch_size * hidden_dim
        :param x_lengths: batch_size
        :return: batch_size * seq_len, batch_size * hidden_dim
        """
        seq_len, batch_size, _ = x.size()   #100,3,32

        flat_inputs = x.reshape(-1, self.hidden_dim)    #[300,32]

        mlp_x = self.proj_w(flat_inputs)    #[300,32]

        # (batch_size, seq_len)
        att_scores = self.proj_v(mlp_x).view(seq_len, batch_size).t()   #[3,998]


        # (seq_len, batch_size)
        normalized_masked_att_scores = masked_softmax(
            att_scores, x_lengths, self.src_length_masking
        ).t()

        # (batch_size, hidden_dim)
        attn_x = (x * normalized_masked_att_scores.unsqueeze(2)).sum(0)

        return normalized_masked_att_scores.t(), attn_x

# s(x, q) = v.T * tanh (W * x + b)
class DL_AttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, attention_dim, src_length_masking=True):
        super(DL_AttentionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.src_length_masking = src_length_masking

        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, x, x_lengths):
        """
        :param x: seq_len * batch_size * hidden_dim
        :param x_lengths: batch_size
        :return: batch_size * seq_len, batch_size * hidden_dim
        """
        seq_len, batch_size, _ = x.size()   #998,3,64

        flat_inputs = x.reshape(-1, self.hidden_dim)    #[2994,64]
        # (seq_len * batch_size, attention_dim)

        mlp_x = self.proj_w(flat_inputs)    #[2994,32]

        # (batch_size, seq_len)
        att_scores = self.proj_v(mlp_x).view(seq_len, batch_size).t()   #[3,998]

        # (seq_len, batch_size)
        normalized_masked_att_scores = masked_softmax(
            att_scores, x_lengths, self.src_length_masking
        ).t()   #[998,3]

        # (batch_size, hidden_dim)
        attn_x = (x * normalized_masked_att_scores.unsqueeze(2)).sum(0)

        return normalized_masked_att_scores.t(), attn_x

# s(x, q) = v.T * tanh (W * x + b)
class Graph_AttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, attention_dim, src_length_masking=True):
        super(Graph_AttentionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.src_length_masking = src_length_masking

        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, x, x_lengths):
        """
        :param x: seq_len * batch_size * hidden_dim
        :param x_lengths: batch_size
        :return: batch_size * seq_len, batch_size * hidden_dim
        """
        seq_len, batch_size, _ = x.size()   #998,3,64

        flat_inputs = x.reshape(-1, self.hidden_dim)    #[2994,64]  [1*3,32]
        # (seq_len * batch_size, attention_dim)

        mlp_x = self.proj_w(flat_inputs)    #[2994,32]

        # (batch_size, seq_len)
        att_scores = self.proj_v(mlp_x).view(seq_len, batch_size).t()   #[3,998]

        # (seq_len, batch_size)
        normalized_masked_att_scores = masked_softmax(
            att_scores, x_lengths, self.src_length_masking
        ).t()   #[998,3]

        # (batch_size, hidden_dim)
        attn_x = (x * normalized_masked_att_scores.unsqueeze(2)).sum(0)

        return normalized_masked_att_scores.t(), attn_x


def create_src_lengths_mask(batch_size: int, src_lengths: Tensor, max_src_len: int):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths
      max_src_len: Optionally override max_src_len for the mask
    Outputs:
      [batch_size, max_src_len]
    """
    src_lengths = torch.tensor(src_lengths)
    src_lengths.to(device)
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)

    src_indices = src_indices.expand(batch_size, max_src_len)

    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)

    # returns [batch_size, max_seq_len]

    '''
    tensor([[1, 0, 0],
           [1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]], dtype=torch.int32)
           需要进行mask的数据
    '''

    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""


    if src_length_masking:
        bsz, max_src_len = scores.size()

        # compute masks
        #src_mask是需要进行mask处理的部分
        src_mask = create_src_lengths_mask(bsz, src_lengths,max_src_len)
        src_mask = src_mask.to(device)

        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -np.inf)

    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
