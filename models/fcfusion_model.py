import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.classifier import FcClassifier
from .networks.fc_encoder import FcEncoder
from .networks.lstm_encoder import LSTMEncoder, BiLSTMEncoder

class FcFusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of lstm')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=256, type=int, help='lstm hidden layer')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--target', default='arousal', type=str, help='one of [arousal, valence]')
        parser.add_argument('--bidirection', default=False, action='store_true', help='whether to use bidirectional lstm')
        parser.add_argument('--normalize', action='store_true', default=False, help='whether to normalize step features')
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = ['MSE']
        self.model_names = ['_fc', '_seq', '_reg']
        
        if opt.hidden_size == -1:
            opt.hidden_size = min(opt.input_dim // 2, 512)
        # net fc fusion
        self.net_fc = FcEncoder(opt.input_dim, [opt.hidden_size, opt.hidden_size], dropout=0.1, dropout_input=False)
        # net seq
        if opt.bidirection:
            self.net_seq = BiLSTMEncoder(opt.hidden_size, opt.hidden_size)
            self.hidden_mul = 2
        else:
            self.net_seq = LSTMEncoder(opt.hidden_size, opt.hidden_size)
            self.hidden_mul = 1
        
        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.hidden_size = opt.hidden_size
        self.net_reg = FcClassifier(opt.hidden_size * self.hidden_mul, layers, 1, dropout=opt.dropout_rate)
        # settings 
        self.target_name = opt.target
        self.max_seq_len = opt.max_seq_len
        if self.isTrain:
            self.criterion_reg = torch.nn.MSELoss(reduction='sum')
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
        self.feature = input['feature']
        self.mask = input['mask'].to(self.device)
        self.length = input['length']
        if self.normalize:
            self.feature = self.normalize_feature(self.feature, self.mask)
        if self.isTrain:
            self.target = input[self.target_name].to(self.device)

    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.feature.size(0)
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = []
        previous_h = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        previous_c = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        for step in range(split_seg_num):
            feature_step = self.feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            prediction, (previous_h, previous_c) = self.forward_step(feature_step, (previous_h, previous_c))
            previous_h = previous_h.detach()
            previous_c = previous_c.detach()
            self.output.append(prediction.squeeze(dim=-1))
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()  
                target = self.target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                self.backward_step(prediction, target, mask)
                self.optimizer.step() 
        self.output = torch.cat(self.output, dim=1)
    
    def forward_step(self, input, states):
        fusion = self.net_fc(input)
        hidden, (h, c) = self.net_seq(fusion, states)
        prediction, _ = self.net_reg(hidden)
        return prediction, (h, c)
   
    def backward_step(self, pred, target, mask):
        """Calculate the loss for back propagation"""
        pred = pred.squeeze() * mask
        target = target * mask
        batch_size = target.size(0)
        self.loss_MSE = self.criterion_reg(pred, target) / batch_size
        self.loss_MSE.backward(retain_graph=False)    
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)
