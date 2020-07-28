import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.classifier import FcClassifier
from .networks.fc_encoder import FcEncoder
from .networks.lstm_encoder import LSTMEncoder, BiLSTMEncoder
from utils.tools import get_dim

class FcMapModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of lstm')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=256, type=int, help='lstm hidden layer')
        parser.add_argument('--affine_dim', default=128, type=int, help='feature affine dim')
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
        self.model_names = ['_seq', '_reg', '_fc']
        self.feature_set = opt.feature_set.split(',')
        self.affine_dim = opt.affine_dim
        
        # net feature_map
        for feature_name in self.feature_set:
            input_dim = get_dim(feature_name)
            net = FcEncoder(input_dim, [self.affine_dim, self.affine_dim], dropout=opt.dropout_rate)
            setattr(self, f'net_{feature_name}', net)
        
        self.model_names += [f'_{feature_name}' for feature_name in self.feature_set]
        # net seq
        if opt.hidden_size == -1:
            opt.hidden_size = (len(self.feature_set)+1) // 2 * self.affine_dim
        
        # net seq
        if opt.bidirection:
            self.net_seq = BiLSTMEncoder(opt.hidden_size, opt.hidden_size)
            self.hidden_mul = 2
        else:
            self.net_seq = LSTMEncoder(opt.hidden_size, opt.hidden_size)
            self.hidden_mul = 1 
        
        # net fc fusion
        self.net_fc = FcEncoder(self.affine_dim*len(self.feature_set), [opt.hidden_size, opt.hidden_size], 
                                dropout=0.2, dropout_input=False)
        
        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.hidden_size = opt.hidden_size
        self.net_reg = FcClassifier(self.hidden_size, layers, 1, dropout=opt.dropout_rate)
        # settings 
        self.target_name = opt.target
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
        self.target = input[self.target_name].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.length = input['length']
        self.feature_lens = input['feature_lens']
        for i, feature_name in enumerate(self.feature_set):
            begin = torch.sum(self.feature_lens[:i]).item()
            end = torch.sum(self.feature_lens[:i+1]).item()
            feature = self.feature[:, :, begin:end]
            setattr(self, feature_name, feature)

    def run(self):
        """After feed a batch of samples, Run the model."""
        # feature affine to latent space using fc layers
        batch_size = self.feature.size(0)
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = []
        previous_h = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        previous_c = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        for step in range(split_seg_num):
            # feature_step = self.feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            affine_feature = []
            for feature_name in self.feature_set:
                net = getattr(self, f'net_{feature_name}')
                feat = getattr(self, feature_name)[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                feat = self.normalize_feature(feat)
                latent_embd = net(feat)
                affine_feature.append(latent_embd)
            affine_feature = torch.cat(affine_feature, dim=-1)

            prediction, (previous_h, previous_c) = self.forward_step(affine_feature, (previous_h, previous_c))
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
    
    def normalize_feature(self, features):
        mean_f = torch.mean(features, dim=1).unsqueeze(1).float()
        std_f = torch.std(features, dim=1).unsqueeze(1).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
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
