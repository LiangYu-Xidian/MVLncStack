from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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

        print(F.softmax(scores.float(), dim=-1).type_as(scores))

    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)