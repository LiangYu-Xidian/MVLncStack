
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from GetAttention import DL_AttentionNetwork, ML_AttentionNetwork, Graph_AttentionNetwork, PositionalEncoding
import sys
from collections import Counter
from torch.nn import functional as F
src_vocab = {'P': 0, 'A': 1, 'G': 2, 'C': 3, 'U': 4}
wordsize = 5
embedingsize = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_dim = 128
hidden_dim = 32

class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()

        self.encode_layer = nn.Embedding(wordsize, embedingsize)

        self.max_kernel = 9
        self.dl_cnn_out = 32
        self.hidden_dim = 16
        self.dl_att_out = 32
        self.num_layers = 1
        self.dl_attention_dim = 32
        self.dropout = 0.1

        self.D_conv = nn.Sequential(
            nn.Conv1d(in_channels=embedingsize, out_channels=2*self.dl_cnn_out, kernel_size=9,),
            nn.Conv1d(in_channels=2*self.dl_cnn_out, out_channels=self.dl_cnn_out,kernel_size=9),
            nn.BatchNorm1d(num_features=self.dl_cnn_out, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.max_kernel),
        )

        self.M_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3,),
            nn.Conv1d(in_channels=64, out_channels=32,kernel_size=3),
            nn.BatchNorm1d(num_features=32, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
        )


        self.pos = PositionalEncoding(d_model=embedingsize, max_len = args.len[0]*2)

        self.bilstm_layer = nn.LSTM(self.dl_cnn_out, self.hidden_dim, self.num_layers, bidirectional=True, batch_first=True)

        self.dl_attention_layer = DL_AttentionNetwork(self.dl_att_out, self.dl_attention_dim)
        self.ml_attention_layer = ML_AttentionNetwork(self.dl_att_out, self.dl_attention_dim)

        self.fc_layer = nn.Linear(self.dl_att_out, 4)
        self.dropout = nn.Dropout(p=self.dropout)



    def forward(self,DL):

        #1.深度学习部分处理特征
        x_encode = self.encode_layer(DL)  # [3,8000,128]
        x_input = x_encode.permute(0,2,1)
        #x_input = x_encode.permute(1, 0, 2)  # [3,128,8000]

        #x_input = self.pos(x_input)  # [6000,3,4]
        #x_input = x_input.permute(1, 2, 0)  # [3,4,6000]

        conved = self.D_conv(x_input)   #[3,32,998]
        conved = conved.permute(0,2,1)  #[3,998,32]
        conved_len = conved.shape[1]    #998
        lengths = []
        for seq in DL:
            leng = int(len(torch.nonzero(seq))/self.max_kernel)
            if leng <= conved_len : lengths.append(leng)
            else: lengths.append(conved_len)
        x_packed_input = pack_padded_sequence(input=conved, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm_layer(x_packed_input)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=conved_len, padding_value=0.0)
        dl_scores, dl_out = self.dl_attention_layer(outputs.permute(1, 0, 2), lengths)

        #深度学习处理后的atten值
        output = self.fc_layer(dl_out)
        output = self.dropout(output)

        return output
