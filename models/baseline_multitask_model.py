import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.classifier import FcClassifier
from .networks.lstm_encoder import LSTMEncoder

class BaselineMultitaskModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of lstm')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=256, type=int, help='lstm hidden layer')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = ['MSE_A', 'MSE_V']
        self.model_names = ['_seq', '_reg_A', '_reg_V']
        # net seq
        if opt.hidden_size == -1:
            opt.hidden_size = opt.input_dim // 2
        self.net_seq = LSTMEncoder(opt.input_dim, opt.hidden_size)
        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.hidden_size = opt.hidden_size
        self.net_reg_A = FcClassifier(opt.hidden_size, layers, 1, dropout=opt.dropout_rate)
        self.net_reg_V = FcClassifier(opt.hidden_size, layers, 1, dropout=opt.dropout_rate)
        # settings 
        self.max_seq_len = opt.max_seq_len
        if self.isTrain:
            self.criterion_reg = torch.nn.MSELoss(reduction='sum')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.feature = input['feature'].to(self.device)
        self.arousal = input['arousal'].to(self.device)
        self.valence = input['valence'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.length = input['length']

    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.arousal.size(0)
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output_a = []
        self.output_v = []
        previous_h = torch.zeros(1, batch_size, self.hidden_size).float().to(self.device) 
        previous_c = torch.zeros(1, batch_size, self.hidden_size).float().to(self.device) 
        for step in range(split_seg_num):
            feature_step = self.feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            pred_a, pred_v, (previous_h, previous_c) = self.forward_step(feature_step, (previous_h, previous_c))
            previous_h = previous_h.detach()
            previous_c = previous_c.detach()
            self.output_a.append(pred_a.squeeze(dim=-1))
            self.output_v.append(pred_v.squeeze(dim=-1))
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()  
                target_a = self.arousal[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                target_v = self.valence[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                self.backward_step(pred_a, pred_v, target_a, target_v, mask)
                self.optimizer.step() 
        self.output_a = torch.cat(self.output_a, dim=1)
        self.output_v = torch.cat(self.output_v, dim=1)
    
    def forward_step(self, input, states):
        hidden, (h, c) = self.net_seq(input, states)
        pred_a, _ = self.net_reg_A(hidden)
        pred_v, _ = self.net_reg_V(hidden)
        return pred_a, pred_v, (h, c)
   
    def backward_step(self, pred_a, pred_v, target_a, target_v, mask):
        """Calculate the loss for back propagation"""
        pred_a = pred_a.squeeze(dim=-1) * mask
        pred_v = pred_v.squeeze(dim=-1) * mask
        target_a = target_a * mask
        target_v = target_v * mask
        assert target_a.size(0) == target_v.size(0)
        batch_size = target_a.size(0)
        self.loss_MSE_A = self.criterion_reg(pred_a, target_a) / batch_size
        self.loss_MSE_V = self.criterion_reg(pred_v, target_v) / batch_size
        loss = self.loss_MSE_A + self.loss_MSE_V
        loss.backward(retain_graph=False)    
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)
