import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class DiDiAttention(nn.Module):
    def __init__(self, input_a, input_l, head=5):
        super(DiDiAttention, self).__init__()
        self.u = nn.Linear(input_a, 1, bias=False)
        self.v = nn.Linear(input_l, 1, bias=True)
        self.eplison = 1e-8
        # self.u = torch.Tensor(input_a, 1)
        # self.v = torch.Tensor(input_l, 1)
        # self.bias = torch.Tensor(1)
        # self.u.requires_grad_(True)
        # self.v.requires_grad_(True)
        # self.bias.requires_grad_(True)
        self.input_a = input_a
        self.input_l = input_l
    
    def calc_atten_weight(self, a, l, len_a, len_l):
        ''' a: A_data of one batch
            l: L_data of one batch
        '''
        a = a[:len_a]
        l = l[:len_l]
        _A = a.unsqueeze(0).expand(len_l, len_a, self.input_a)
        _L = l.unsqueeze(1).expand(len_l, len_a, self.input_l)
        attention_matrix = torch.tanh(self.u(_A) + self.v(_L))      # (len_l, len_a, 1)
        norm_vector = torch.sum(attention_matrix, dim=1)            # (len_l, 1)
        attention_matrix = attention_matrix.squeeze() / norm_vector # formular (8) in paper
        return attention_matrix.squeeze()

    def forward(self, A, L, length_a, length_l):
        batch_size = A.size(0)
        seq_len_l = L.size(1)
        seq_len_a = A.size(1)

        attention_matrix = []
        attention_matrix = list(map(
            lambda x: self.calc_atten_weight(*x), zip(A, L, length_a, length_l)
        ))
        ans = []
        for batch_idx in range(batch_size):
            k = A[batch_idx, :length_a[batch_idx], :]
            att_feat = torch.sum(
                A[batch_idx, :length_a[batch_idx], :].unsqueeze(0).expand(length_l[batch_idx], length_a[batch_idx], self.input_a) *
                attention_matrix[batch_idx].unsqueeze(-1), # (len_l, len_a, hidden_size) * (len_l, len_a, 1)
                dim=1
            )
            ans.append(att_feat)

        ans = pad_sequence(ans, batch_first=True)
        return ans
    
    # def calc_atten_weight(self, A, L):
    #     batch_size = A.size(0)
    #     seq_len_l = L.size(1)
    #     seq_len_a = A.size(1)
    #     _A = A.unsqueeze(1).expand(batch_size, seq_len_l, seq_len_a, self.input_a)
    #     _L = L.unsqueeze(2).expand(batch_size, seq_len_l, seq_len_a, self.input_l)
    #     # attention_matrix = F.tanh(_A @ self.u + _L @ self.v + self.bias)
    #     attention_matrix = torch.tanh(self.u(_A) + self.v(_L))
    #     return attention_matrix.squeeze()
    
    # def forward(self, A, L, mask):
    #     ''' A.size() => [batch_size, seq_len_a, hidden_size]
    #         L.size() => [batch_size, seq_len_l, hidden_size]
    #         length_a: (batch_size, ), lengths for A input
    #         length_l: (batch_size, ), lengths for L input
    #     '''
    #     seq_len_l = L.size(1)
    #     seq_len_a = A.size(1)
    #     batch_size = A.size(0)
    #     attention_matrix = self.calc_atten_weight(A, L)
    #     attention_matrix = torch.exp(attention_matrix) * mask
    #     norm_vector = torch.sum(attention_matrix, dim=1).unsqueeze(1).expand(attention_matrix.size()) + self.eplison
    #     attention_matrix = attention_matrix / norm_vector
    #     # sum_j{a_ji * s_i} => ans with shapt [batch_size, seq_len_l, hidden_size_a]
    #     ans = torch.sum(
    #         attention_matrix.unsqueeze(-1).expand(batch_size, seq_len_l, seq_len_a, self.input_a) *
    #         A.unsqueeze(1).expand(batch_size, seq_len_l, seq_len_a, self.input_a), 
    #         dim=1
    #     )

    #     return ans
