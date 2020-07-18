import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2) 
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / (bandwidth_temp + 1e-6) ) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        # print("FIRST",source[0][:5], target[0][:5])
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        
        return loss

class KLDivLoss_OnFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.KLDivLoss()
    
    def forward(self, feat1, feat2):
        # feat1 = F.log_softmax(feat1, dim=-1)
        # feat2 = F.softmax(feat2, dim=-1)
        return self.loss(feat1, feat2)

class SoftCenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Change it to a soft label version.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=4, feat_dim=128, use_gpu=True):
        super(SoftCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        init_tensor = torch.Tensor(self.num_classes, self.feat_dim)
        torch.nn.init.xavier_normal_(init_tensor)
        # whether use tensor.cuda()
        if self.use_gpu:
            self.centers = nn.Parameter(init_tensor).cuda()
        else:
            self.centers = nn.Parameter(init_tensor)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: soft label with shape (batch_size, class_num).
        """
        batch_size, feat_dim = x.size()
        # x expand to size [class_num, batch_size, feat_dim]
        x_mat = x.expand(self.num_classes, batch_size, feat_dim)
        # center expand to size [class_num, batch_size, feat_dim]
        center_mat = self.centers.unsqueeze(1).expand(self.num_classes, batch_size, feat_dim)
        # calculate the square error using "torch.sum((x_mat - center_mat)**2, dim=-1)"
        # and then do the weighted sum using weight labels
        delta = torch.sum((x_mat - center_mat)**2, dim=-1) * labels.t()
        loss = torch.sum(delta) / batch_size
        return loss


class SpectralLoss(nn.Module):
    ''' Calculate spectral loss 
        L_{spec} = mean(wij * ||Yi - Yj||_2) for each pair in a mini-batch.
    '''
    def __init__(self, adjacent):
        super().__init__()
        self.adjacent = torch.from_numpy(adjacent).cuda().float()
        self.epsilon = 1e-6

    def forward(self, batch_data, batch_indexs):
        ''' batch_data: [batch_size, feat_dim]
        '''
        batch_size = batch_data.size(0)
        feat_dim = batch_data.size(1)
        ai = batch_data.expand(batch_size, batch_size, feat_dim)
        aj = ai.transpose(0, 1)
        local_adjacent = self.adjacent[batch_indexs][:, batch_indexs]
        loss = torch.sum(torch.sqrt(torch.sum((ai-aj)**2, dim=2) + self.epsilon) * local_adjacent)
        return loss / (batch_size * batch_size)

        # batch_size = batch_data.size(0)
        # feat_dim = batch_data.size(1)
        # local_adjacent = self.adjacent[batch_indexs][:, batch_indexs]
        # total_loss = torch.as_tensor(0.0).cuda()
        # for i in range(batch_size):
        #     for j in range(batch_size):
        #         weight = local_adjacent[i, j]
        #         total_loss += weight * torch.dist(batch_data[i], batch_data[j], p=2)
        # return total_loss / (batch_size * batch_size)

class OrthPenalty(nn.Module):
    ''' Calculate orth penalty
        if input batch of feat is Y with size [batch_size, feat_dim]
        L_{orth} = sum(|Y@Y.T - I|) / batch_size**2 
                   where I is a diagonal matrix with size [batch_size, batch_size]
    '''
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6
    
    def forward(self, batch_data):
        ''' batch_data: [batch_size, feat_dim]
        '''
        batch_size = batch_data.size(0)
        feat_dim = batch_data.size(1)
        I = torch.eye(feat_dim).cuda() * batch_size
        loss = torch.sum(torch.sqrt(((batch_data.transpose(0, 1) @ batch_data) - I + self.epsilon)**2))
        return loss / (batch_size) 
       

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq + 1e-6)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss