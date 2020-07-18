import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class GruEncoder(nn.Module):
    ''' Gru encoder
    '''
    def __init__(self, input_size, hidden_size, embd_size):
        super(GruEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_size = embd_size
        self.rnn = nn.GRU(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.embd_size),
            nn.ReLU(),
        )

    def forward(self, x, length):
        batch_size = x.size(0)
        # x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        _, sequence_embd = self.rnn(x)
        sequence_embd = sequence_embd.view(batch_size, -1)
        embd = self.fc(sequence_embd)
        return embd