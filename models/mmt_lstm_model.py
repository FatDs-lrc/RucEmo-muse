import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.classifier import FcClassifier
from .networks.mmt_lstm import MMTLstm

class MMTLstmModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of lstm')
        parser.add_argument('--hidden_size', default=60, type=int, help='lstm hidden layer')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--target', default='arousal', type=str, help='one of [arousal, valence]')
        parser.add_argument('--bidirectional', default=False, action='store_true', help='whether to use bidirectional lstm')
        parser.add_argument('--normalize_a', action='store_true', default=False, help='whether to normalize step features')
        parser.add_argument('--normalize_v', action='store_true', default=False, help='whether to normalize step features')
        parser.add_argument('--normalize_l', action='store_true', default=False, help='whether to normalize step features')
        
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = ['MSE']
        self.model_names = ['_seq', '_reg']
        # net seq
        if opt.hidden_size == -1:
            opt.hidden_size = min(opt.input_dim // 2, 512)
            opt.hidden_size += opt.hidden_size % opt.num_heads
        
        self.net_seq = MMTLstm(opt.a_dim, opt.v_dim, opt.l_dim, 
                            opt.hidden_size, bidirectional=opt.bidirectional)

        self.hidden_mul = 2 if opt.bidirectional else 1
        self.hidden_size = opt.hidden_size

        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.hidden_size = opt.hidden_size
        self.net_reg = FcClassifier(opt.hidden_size * self.hidden_mul, layers, 1, dropout=opt.dropout_rate)
        
        # settings 
        self.target_name = opt.target
        self.max_seq_len = opt.max_seq_len
        if self.isTrain:
            self.criterion_reg = torch.nn.MSELoss(reduction='sum')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
        self.normalize_a = opt.normalize_a
        self.normalize_v = opt.normalize_v
        self.normalize_l = opt.normalize_l
    
    def normalize_feature(self, features):
        mean_f = torch.mean(features, dim=1).unsqueeze(1).float()
        std_f = torch.std(features, dim=1).unsqueeze(1).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    def set_input(self, input, load_label=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.a_feature = input['a_feature'].to(self.device)
        self.v_feature = input['v_feature'].to(self.device)
        self.l_feature = input['l_feature'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.length = input['length']
        if self.normalize_a:
            self.a_feature = self.normalize_feature(self.a_feature)
        if self.normalize_v:
            self.v_feature = self.normalize_feature(self.v_feature)
        if self.normalize_l:
            self.l_feature = self.normalize_feature(self.l_feature)
        if load_label:
            self.target = input[self.target_name].to(self.device)


    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.a_feature.size(0)
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        A_previous_h = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        A_previous_c = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device)
        V_previous_h = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        V_previous_c = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device)
        L_previous_h = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        L_previous_c = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device)
        F_previous_h = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        F_previous_c = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device)
        
        self.output = [] 
        for step in range(split_seg_num):
            a_feature_step = self.a_feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            v_feature_step = self.v_feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            l_feature_step = self.l_feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            previous_states = (A_previous_h, A_previous_c, V_previous_h, V_previous_c, L_previous_h, L_previous_c, F_previous_h, F_previous_c)
            prediction, cur_states = self.forward_step(a_feature_step, v_feature_step, l_feature_step, previous_states)
            A_previous_h, A_previous_c, V_previous_h, V_previous_c, \
            L_previous_h, L_previous_c, F_previous_h, F_previous_c = cur_states
            self.output.append(prediction.squeeze(dim=-1))
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()  
                target = self.target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                self.backward_step(prediction, target, mask)
                self.optimizer.step() 
        self.output = torch.cat(self.output, dim=1)
    
    def forward_step(self, a_data, v_data, l_data, previous_states):
        A_previous_h, A_previous_c, V_previous_h, V_previous_c, \
        L_previous_h, L_previous_c, F_previous_h, F_previous_c = previous_states
        state_l = (L_previous_h, L_previous_c)
        state_a = (A_previous_h, A_previous_c)
        state_v = (V_previous_h, V_previous_c)
        state_f = (F_previous_h, F_previous_c)

        hidden, cur_states = self.net_seq(a_data, v_data, l_data, state_a, state_v, state_l, state_f)
        prediction, _ = self.net_reg(hidden)
        cur_states = (x.detach() for x in cur_states)
        return prediction, cur_states
   
    def backward_step(self, pred, target, mask):
        """Calculate the loss for back propagation"""
        pred = pred.squeeze() * mask
        target = target * mask
        batch_size = target.size(0)
        self.loss_MSE = self.criterion_reg(pred, target) / batch_size
        self.loss_MSE.backward(retain_graph=False)    
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)
