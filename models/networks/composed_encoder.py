import torch
import torch.nn as nn
import math
from models.networks.mmt_modules.transformer import TransformerEncoder

class LstmTransformer(nn.Module):
    ''' Feature firstly input to a lstm module and then send into a transformer encoder
        Note that TransformerEncoder has input shape of [seq_len, batch_size, embd_size]
        This module has input shape [batch_size, seq_len, embd_size]
    '''
    def __init__(self, input_dim, hidden_size, num_heads, num_layers):
        super(LstmTransformer, self).__init__()
        self.attn_dropout = 0.1
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = 0.25
        self.attn_mask = False
        self.rnn_dropout = 0.1
        self.rnn = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.transformer = TransformerEncoder(hidden_size, num_heads, num_layers, 
                        attn_dropout=self.attn_dropout, relu_dropout=self.relu_dropout,
                        res_dropout=self.res_dropout, embed_dropout=self.embed_dropout,
                        attn_mask=self.attn_mask)
        self.dropout = nn.Dropout(self.rnn_dropout)
    
    def forward(self, x, states):
        r_out, (h_n, h_c) = self.rnn(x, states)
        r_out = self.dropout(r_out).transpose(0, 1)
        out = self.transformer(r_out).transpose(0, 1)
        return out, (h_n, h_c)

class TransformerLstm(nn.Module):
    ''' Feature firstly input to a transformer encoder and then send into a lstm module
        Note that TransformerEncoder has input shape of [seq_len, batch_size, embd_size]
        This module has input shape [batch_size, seq_len, embd_size]
    '''
    def __init__(self, input_dim, hidden_size, num_heads, num_layers):
        super(TransformerLstm, self).__init__()
        self.attn_dropout = 0.1
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = 0.25
        self.attn_mask = False
        self.mid_dropout = 0.3
        self.conv = nn.Conv1d(input_dim, hidden_size, kernel_size=1, padding=0, bias=False)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.transformer = TransformerEncoder(hidden_size, num_heads, num_layers, 
                        attn_dropout=self.attn_dropout, relu_dropout=self.relu_dropout,
                        res_dropout=self.res_dropout, embed_dropout=self.embed_dropout,
                        attn_mask=self.attn_mask)
        self.dropout = nn.Dropout(self.mid_dropout)
    
    def forward(self, x, states):
        x = self.conv(x.transpose(1, 2))
        x = x.permute(2, 0, 1)
        attn_out = self.transformer(x).transpose(0, 1)
        attn_out = self.dropout(attn_out)
        r_out, (h_n, h_c) = self.rnn(attn_out, states)
        r_out = self.dropout(r_out)
        return r_out, (h_n, h_c)
    
if __name__ == '__main__':
    # module = LstmTransformer(300, 256, 4, 2)
    module = TransformerLstm(300, 256, 4, 2)
    x = torch.rand(10, 128, 300)
    out, _ = module(x)
    print(out.shape)


# from models.utils.positional_encoding import get_sinusoid_encoding_table

# class TransformerEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_size, num_heads=4, num_layers=3):
#         super(TransformerEncoder, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         # self.attn_affine = Attention(self.input_dim, self.hidden_size)
#         self.pos_emb = nn.Embedding.from_pretrained(
#             get_sinusoid_encoding_table(1024, hidden_size, padding_idx=0),
#             freeze=True
#         )
#         self.pos_dropout = nn.Dropout(p=0.1)
#         assert self.hidden_size % self.num_heads == 0, 'hidden size must be divisible by num_heads'
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.num_heads)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
    
#     def forward(self, x, mask=None):
#         """ In torch 1.5.0 this block does't support batch_first optional parameter"""
#         """ Since our input in batch first, tranpose is in need """
#         batch_size, seq_len, embd_dim = x.size()
#         src_pos = torch.arange(seq_len).to(x).long() + 1
#         src_pos = src_pos.unsqueeze(0).expand(batch_size, seq_len)
#         if mask is not None:
#             src_pos = src_pos * mask.long()
#             mask = (1-mask).byte()
            
#         pos = self.pos_emb(src_pos) / 10
#         x = self.pos_dropout(x + pos)
#         # x = self.attn_affine(x, mask)
#         x = x.transpose(0, 1)
#         out = self.encoder(x, src_key_padding_mask=mask)
#         # print(out)
#         return out.transpose(0, 1)
    
# class Attention(nn.Module):
#     def __init__(self, input_dim, output_dim, fc_dim=512, bias=True):
#         super(Attention, self).__init__()
#         self.output_dim = output_dim
#         self.query_proj = nn.Linear(input_dim, output_dim, bias=bias)
#         self.key_proj = nn.Linear(input_dim, output_dim, bias=bias)
#         self.value_proj = nn.Linear(input_dim, output_dim, bias=bias)
#         self.fc = nn.Sequential(
#             nn.Linear(output_dim, fc_dim), 
#             nn.Linear(fc_dim, output_dim), 
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#         self.layer_norm = nn.LayerNorm(output_dim)
    
#     def forward(self, x, key_mask=None, query_mask=None):
#         ''' calc attention acourding to z = softmax(frac^{Q*K^T}_{sqrt(d_k)}) * V
#             x: shape [batch_size, seq_len, embd_dim]
#             mask: shape [batch_size, seq_len], with padding set to 0

#         '''
#         query = self.query_proj(x)
#         key = self.key_proj(x)
#         value = self.value_proj(x)
        
#         scaling = float(self.output_dim) ** -0.5
#         query = query * scaling
        
#         attn = torch.softmax(torch.bmm(query, key.transpose(-1, -2)), dim=-1)
#         if key_mask is not None:
#             attn_key_mask = key_mask.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
#             attn = attn.masked_fill(key_mask, -2 ** 32 + 1)

#         if query_mask is not None:
#             attn = attn * query_mask.unsqueeze(-1).repeat(1, 1, x.size(1))
        
#         z = torch.bmm(attn, value)
#         z = self.layer_norm(self.fc(z))
#         return z

# if __name__ == '__main__':
#     # te = TransformerEncoder(314, 512)
#     # print(te)
#     # src = torch.rand(10, 32, 314)
#     # out = te(src)
#     # print(out.shape)
#     # attn = Attention(314, 512, bias=True)
#     # src = torch.rand(10, 32, 314)
#     # out = attn(src)
#     # print(out.shape)
#     a = nn.Embedding.from_pretrained(
#         get_sinusoid_encoding_table(1024, 100, padding_idx=0),
#         freeze=True
#     )
#     x = torch.rand(1, 10, 100)
#     pos = torch.arange(x.size(1)) + 1
#     pos = pos.unsqueeze(0)
#     pos = pos.expand(4, pos.size(-1))
#     pos = a(pos)
#     print(pos.shape)