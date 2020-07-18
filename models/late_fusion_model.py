import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.classifier import FcClassifier
from .networks.lstm_encoder import LSTMEncoder, BiLSTMEncoder

class LateFusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of lstm')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=60, type=int, help='lstm hidden layer')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--target', default='arousal', type=str, help='one of [arousal, valence]')
        parser.add_argument('--bidirection', default=False, action='store_true', help='whether to use bidirectional lstm')
        parser.add_argument('--normalize', action='store_true', default=False, help='whether to normalize step features')
        
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
            Using MuseWildSplitDataset
        """
        super().__init__(opt, logger)
        self.loss_names = ['MSE']
        self.model_names = ['A_seq', 'V_seq', 'L_seq', '_reg']
        # net seq
        if opt.hidden_size == -1:
            self.a_hidden_size = min(opt.a_dim // 2, 512)
            self.v_hidden_size = min(opt.v_dim // 2, 512)
            self.l_hidden_size = min(opt.l_dim // 2, 512)

        if opt.bidirection:
            self.netA_seq = BiLSTMEncoder(opt.a_dim, self.a_hidden_size)
            self.netV_seq = BiLSTMEncoder(opt.v_dim, self.v_hidden_size)
            self.netL_seq = BiLSTMEncoder(opt.l_dim, self.l_hidden_size)
            self.hidden_mul = 2
        else:
            self.netA_seq = LSTMEncoder(opt.a_dim, self.a_hidden_size)
            self.netV_seq = LSTMEncoder(opt.v_dim, self.v_hidden_size)
            self.netL_seq = LSTMEncoder(opt.l_dim, self.l_hidden_size)
            self.hidden_mul = 1

        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.hidden_size = self.a_hidden_size + self.v_hidden_size + self.l_hidden_size
        self.net_reg = FcClassifier(self.hidden_size * self.hidden_mul, layers, 1, dropout=opt.dropout_rate)

        # settings 
        self.target_name = opt.target
        self.max_seq_len = opt.max_seq_len
        if self.isTrain:
            self.criterion_reg = torch.nn.MSELoss(reduction='sum')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
        self.normalize = opt.normalize
    
    def normalize_feature(self, features, mask):
        mean_f = torch.mean(features, dim=1).unsqueeze(1).float()
        std_f = torch.std(features, dim=1).unsqueeze(1).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.a_feature = input['a_feature'].to(self.device)
        self.v_feature = input['v_feature'].to(self.device)
        self.l_feature = input['l_feature'].to(self.device)
        assert self.a_feature.size(1) == self.v_feature.size(1) == self.l_feature.size(1)
        self.mask = input['mask'].to(self.device)
        self.length = input['length']
        if self.normalize:
            self.a_feature = self.normalize_feature(self.a_feature, self.mask)
            self.v_feature = self.normalize_feature(self.v_feature, self.mask)
            self.l_feature = self.normalize_feature(self.l_feature, self.mask)

        if self.isTrain:
            self.target = input[self.target_name].to(self.device)

    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.a_feature.size(0)
        # batch_max_length = torch.max(self.length).item()
        batch_max_length = self.a_feature.size(1)
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = []
        A_previous_h = torch.zeros(self.hidden_mul, batch_size, self.a_hidden_size).float().to(self.device) 
        A_previous_c = torch.zeros(self.hidden_mul, batch_size, self.a_hidden_size).float().to(self.device)
        V_previous_h = torch.zeros(self.hidden_mul, batch_size, self.v_hidden_size).float().to(self.device) 
        V_previous_c = torch.zeros(self.hidden_mul, batch_size, self.v_hidden_size).float().to(self.device)
        L_previous_h = torch.zeros(self.hidden_mul, batch_size, self.l_hidden_size).float().to(self.device) 
        L_previous_c = torch.zeros(self.hidden_mul, batch_size, self.l_hidden_size).float().to(self.device)

        for step in range(split_seg_num):
            a_feature_step = self.a_feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            v_feature_step = self.v_feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            l_feature_step = self.l_feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            previous_states = (A_previous_h, A_previous_c, V_previous_h, V_previous_c, L_previous_h, L_previous_c)
            prediction, cur_states = self.forward_step(a_feature_step, v_feature_step, l_feature_step, previous_states)
            A_previous_h, A_previous_c, V_previous_h, V_previous_c, L_previous_h, L_previous_c = cur_states
            self.output.append(prediction.squeeze(dim=-1))
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()  
                target = self.target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                self.backward_step(prediction, target, mask)
                self.optimizer.step() 
        self.output = torch.cat(self.output, dim=1)
    
    def forward_step(self, a_data, v_data, l_data, previous_states):
        A_previous_h, A_previous_c, V_previous_h, V_previous_c, L_previous_h, L_previous_c = previous_states
        A_states = (A_previous_h, A_previous_c)
        V_states = (V_previous_h, V_previous_c)
        L_states = (L_previous_h, L_previous_c)
        hidden_a, (h_a, c_a) = self.netA_seq(a_data, A_states)
        hidden_v, (h_v, c_v) = self.netV_seq(v_data, V_states)
        hidden_l, (h_l, c_l) = self.netL_seq(l_data, L_states)
        hidden = torch.cat([hidden_a, hidden_v, hidden_l], dim=-1)
        prediction, _ = self.net_reg(hidden)
        return prediction, (h_a.detach(), c_a.detach(), h_v.detach(), c_v.detach(), h_l.detach(), c_l.detach())
   
    def backward_step(self, pred, target, mask):
        """Calculate the loss for back propagation"""
        pred = pred.squeeze() * mask
        target = target * mask
        batch_size = target.size(0)
        self.loss_MSE = self.criterion_reg(pred, target) / batch_size
        self.loss_MSE.backward(retain_graph=False)    
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)
