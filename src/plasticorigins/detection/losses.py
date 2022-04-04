import torch
from torch.nn.functional import l1_loss as torch_l1_loss
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, train=True, centernet_output=True):
        super(FocalLoss, self).__init__()
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.focal_loss = self._focal_loss if train else self._focal_loss_class_wise
        self.loss = self._train_loss if train else self._test_loss
        self.centernet_output = centernet_output

    def forward(self, pred, gt):

        return self.loss(pred, gt)

    def _train_loss(self, pred, gt):

        if self.centernet_output: pred = pred[0]

        pred_centers = pred['hm']
        gt_centers = gt[:,:-2,:,:]

        # pred_wh = pred['wh']
        # gt_wh = gt[:,-2:,:,:]        

        pos_inds = gt_centers.eq(1).float()
        neg_inds = gt_centers.lt(1).float()

        return self.focal_loss(pred_centers, gt_centers, pos_inds, neg_inds) #+ 0.1 * _l1_loss(pred_wh, gt_wh, pos_inds)

    def _test_loss(self, pred, gt):
        
        if self.centernet_output: pred = pred[0]

        pred_centers = pred['hm']
        gt_centers = gt[:,:-2,:,:]

        pos_inds = gt_centers.eq(1).float()
        neg_inds = gt_centers.lt(1).float()

        return self._focal_loss_class_wise(pred_centers, gt_centers, pos_inds, neg_inds)

    def _focal_loss(self, pred_hm, gt_hm, pos_inds, neg_inds):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''

        pred_hm = _sigmoid(pred_hm)

        neg_weights = torch.pow(1 - gt_hm, self.beta)

        loss = 0.0

        pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos 
            
        return loss

    def _focal_loss_class_wise(self, pred_hm, gt_hm, pos_inds, neg_inds):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''

        pred_hm = _sigmoid(pred_hm)

        neg_weights = torch.pow(1 - gt_hm, self.beta)


        pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * neg_weights * neg_inds

        num_pos  = torch.sum(pos_inds.float(), dim=(0,2,3))
        pos_loss = torch.sum(pos_loss, dim=(0,2,3))
        neg_loss = torch.sum(neg_loss, dim=(0,2,3))

        loss = torch.zeros(size=(len(num_pos),))

        for i, num_poss_class in enumerate(num_pos):

            if num_poss_class == 0:
                loss[i] = loss[i] - neg_loss[i]
            else:
                loss[i] = loss[i] - (pos_loss[i] + neg_loss[i]) / num_poss_class

        return loss

def _l1_loss(pred, gt, pos_inds):

    pos_inds = torch.sum(pos_inds, axis=1).ge(1).unsqueeze(1).expand_as(pred)
    return torch_l1_loss(pred[pos_inds], gt[pos_inds], size_average=False) / (pos_inds.float().sum() + 1e-4)

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y
