import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        # self.bn = nn.BatchNorm1d(self.hidden_size * 2)
        self.layer_norm = nn.LayerNorm((self.hidden_size*2, ))
        
    def forward(self, sequence, lengths):
        self.rnn.flatten_parameters()
        packed_sequence = pack_padded_sequence(sequence, lengths, enforce_sorted=False)
        packed_h, (final_h, _) = self.rnn(packed_sequence)
        h, _ = pad_packed_sequence(packed_h, batch_first=True)
       
        batch_size, max_seq_len = h.size(0), h.size(1)
        out_sequence = []
        for batch_idx in range(batch_size):
            raw_length = lengths[batch_idx]
            feat = torch.cat([
                h[batch_idx, :raw_length, :self.hidden_size],
                h[batch_idx, range(raw_length-1, -1, -1), self.hidden_size:]
            ], dim=-1)
            out_sequence.append(feat)
        
        # out_sequence_lens = torch.tensor([feat.size(0) for feat in out_sequence]).long().cuda()
        out_sequence = pad_sequence(out_sequence, batch_first=True)
        # out_sequence = self.layer_norm(out_sequence)
        return out_sequence
