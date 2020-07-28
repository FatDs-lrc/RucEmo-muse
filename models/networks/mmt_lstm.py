import torch
import math
from torch import nn
import torch.nn.functional as F

from .mmt_modules.transformer import TransformerEncoder
from .lstm_encoder import FcLstmEncoder


class LstmMULTModel(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, hidden_size, num_heads, num_layers, bidirectional=False):
        """
        Construct a MulT model.
        """
        super(LstmMULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = l_dim, a_dim, v_dim
    
        if bidirectional:
            self.d_l, self.d_a, self.d_v = hidden_size*2, hidden_size*2, hidden_size*2
        else:
            self.d_l, self.d_a, self.d_v = hidden_size, hidden_size, hidden_size

        self.vonly = True
        self.aonly = True
        self.lonly = True
        self.num_heads = num_heads
        self.layers = num_layers
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0
        self.attn_dropout_v = 0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = 0.1
        self.attn_mask = True

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        output_dim = 1  

        # 1. Temporal convolutional layers
        self.proj_l = nn.LSTM(self.orig_d_l, hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=1)
        self.proj_a = nn.LSTM(self.orig_d_a, hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=1)
        self.proj_v = nn.LSTM(self.orig_d_v, hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=1)
        
        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_l, x_a, x_v, state_l, state_a, state_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)
        x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)
        x_v = F.dropout(x_v, p=self.embed_dropout, training=self.training)
       
        # Project the textual/visual/audio features
        proj_x_l, (h_l, c_l) = self.proj_l(x_l, state_l)
        proj_x_a, (h_a, c_a) = self.proj_a(x_a, state_a)
        proj_x_v, (h_v, c_v) = self.proj_v(x_v, state_v)

        proj_x_a = proj_x_a.permute(1, 0, 2)
        proj_x_v = proj_x_v.permute(1, 0, 2)
        proj_x_l = proj_x_l.permute(1, 0, 2)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            # last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            # last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            # last_h_v = last_hs = h_vs[-1]
        
        if self.partial_mode == 3:
            # last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
            hs = torch.cat([h_ls, h_as, h_vs], dim=-1)
            hs = hs.permute(1, 0, 2)
        
        # A residual block
        # last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        # last_hs_proj += last_hs
        hs_proj = self.proj2(F.dropout(F.relu(self.proj1(hs)), p=self.out_dropout, training=self.training))
        hs_proj += hs
        
        output = self.out_layer(hs_proj)
        return output, hs, (h_l, c_l, h_a, c_a, h_v, c_v)

class MMTLstm(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, hidden_size, bidirectional=False):
        """
        Construct a MulT model.
        """
        super(MMTLstm, self).__init__()
        self.a_proj = nn.LSTM(a_dim, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.v_proj = nn.LSTM(v_dim, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.l_proj = nn.LSTM(l_dim, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.c_proj = FcLstmEncoder(a_dim+v_dim+l_dim, hidden_size * 3, bidirectional=bidirectional)
        hidden_mul = 1 if not bidirectional else 2
        self.fc1_a = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(hidden_size*3*hidden_mul, hidden_size),
            nn.ReLU()
        )
        self.fc1_v = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(hidden_size*3*hidden_mul, hidden_size),
            nn.ReLU()
        )
        self.fc1_l = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(hidden_size*3*hidden_mul, hidden_size),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden_size*3, hidden_size*2),
            nn.ReLU()
        )
        self.fusion_rnn = nn.LSTM(2*hidden_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.scale = math.sqrt(hidden_size * hidden_mul)

    def forward(self, a_feat, v_feat, l_feat, a_states, v_states, l_states, f_states):
        ''' Input shape [batch_size, seq_len, dim]
        '''
        _a_feat, a_states = self.a_proj(a_feat, a_states)
        _v_feat, v_states = self.v_proj(v_feat, l_states)
        _l_feat, l_states = self.l_proj(l_feat, v_states)
        _c_feat, c_states = self.c_proj(torch.cat([a_feat, v_feat, l_feat], dim=-1))
        attn_a = self.fc1_a(self.get_attn(_a_feat, _v_feat, _l_feat)) + _a_feat
        attn_v = self.fc1_v(self.get_attn(_v_feat, _a_feat, _l_feat)) + _v_feat
        attn_l = self.fc1_l(self.get_attn(_l_feat, _a_feat, _v_feat)) + _l_feat
        fusion = torch.cat([attn_a, attn_v, attn_l])
        fusion = self.dropout(self.fc2(fusion), dim=-1)))
        fusion, f_states = self.fusion_rnn(fusion, f_states)
        return fusion, (*a_states, *v_states, *l_states, *f_states)
    
    def get_attn(self, target_feat, aux_feat1, aux_feat2):
        t_from_t = self.attn(target_feat, target_feat, target_feat)
        t_from_1 = self.attn(aux_feat1, target_feat, target_feat)
        t_from_2 = self.attn(aux_feat2, target_feat, target_feat)
        attn_feat = torch.cat([t_from_t, t_from_1, t_from_2], dim=-1)
        return attn_feat

    def attn(self, query, key, value):
        attn_weight = torch.bmm(query, key.transpose(2, 1))
        attn_weight = F.softmax(attn_weight / self.scale, dim=-1)
        ret = torch.bmm(attn_weight, value)
        return ret

if __name__ == '__main__':
    a = torch.rand(2, 10, 100)
    v = torch.rand(2, 10, 110)
    l = torch.rand(2, 10, 120)
    state_a = (torch.zeros(2, 2, 128), torch.zeros(2, 2, 128))
    state_v = (torch.zeros(2, 2, 128), torch.zeros(2, 2, 128))
    state_l = (torch.zeros(2, 2, 128), torch.zeros(2, 2, 128))
    model = MMTLstm(100, 110, 120, 128, bidirectional=True)
    print(model)
    hidden, states = model(a, v, l, state_l, state_a, state_l)
    print(hidden.size())
