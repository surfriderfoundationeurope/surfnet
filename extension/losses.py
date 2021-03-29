import torch
import torch.nn as nn
from torch import sigmoid
import math 
import matplotlib.pyplot as plt

class TrainLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, sigma2=2):
        super(TrainLoss, self).__init__()
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.sigma2 = int(sigma2)

    def forward(self, h0, h1, Phi0, Phi1):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''


        pos_loss_0, neg_loss_0, num_pos_0 = _pixel_wise_focal_loss(h0, Phi0, self.alpha, self.beta)

        pos_loss_0 = pos_loss_0.sum()
        neg_loss_0 = neg_loss_0.sum()


        pos_loss_1, neg_loss_1, num_pos_1 = _pixel_wise_focal_loss(h1, Phi1, self.alpha, self.beta)

        relevant_pixels_1 = h1.ne(-50).float()

        pos_loss_1 = (pos_loss_1*relevant_pixels_1).sum()
        neg_loss_1 = (neg_loss_1*relevant_pixels_1).sum()


        loss = 0.0 

        if num_pos_0 == 0:
            loss -= 0.5*neg_loss_0
        else:
            loss -= 0.5*(pos_loss_0 + neg_loss_0) / num_pos_0
            
        if num_pos_1 == 0:
            loss -= 0.5*neg_loss_1
        else:
            loss -= 0.5*(pos_loss_1 + neg_loss_1) / num_pos_1

        
        # with open('verbose_hardcore.pickle','wb') as f:
        #     import pickle
        #     obj = (fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6))
        #     pickle.dump(obj, f)
        # plt.close()           
        
        # fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(3,3, figsize=(20,20))

        # ax0.imshow(torch.sigmoid(logit_Phi0[0][0]).cpu(),cmap='gray')
        # ax0.set_title('phi0')

        # ax1.imshow(logit_Phi0[0][0].cpu(),cmap='gray')
        # ax1.set_title('logitphi0')

        # ax2.imshow(h0[0][0].detach().cpu(),cmap='gray')
        # ax2.set_title('h0')

        # ax3.imshow(neg_weights_0[0][0].detach().cpu(),cmap='gray')
        # ax3.set_title('neg_weights')

        # ax4.imshow(neg_loss_0_focal_term[0][0].detach().cpu(), cmap='gray')
        # ax4.set_title('neg_loss_focal_term')

        # ax5.imshow(pos_loss_0_focal_term[0][0].detach().cpu(), cmap='gray')
        # ax5.set_title('pos_loss_focal_term')

        # ax6.imshow(term0[0][0].detach().cpu(), cmap='gray')
        # ax6.set_title('term0')

        # ax7.imshow((pos_loss_0_focal_term * term0)[0][0].detach().cpu(), cmap='gray')
        # ax7.set_title('pos_loss')

        # ax8.imshow((neg_loss_0_focal_term * term0)[0][0].detach().cpu(), cmap='gray')
        # ax8.set_title('neg_loss')
        # plt.suptitle('Loss: {}'.format(loss))
        # plt.show()



        # if math.isnan(loss):
        #     test = 0

        return loss     

class TrainLossOneTerm(nn.Module):
    def __init__(self, alpha=2, beta=4, sigma2=2):
        super(TrainLossOneTerm, self).__init__()
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.sigma2 = int(sigma2)

    def forward(self, h0, logit_Phi0):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''

        # h0 = logit_Phi0.clone()
        # h1 = logit_Phi1.clone()

        # shape_data = logit_Phi0.numel()

        logit_of_clamped_one = torch.logit(torch.tensor(1-1e-16,  dtype=logit_Phi0.dtype))

        pos_inds_0 = logit_Phi0.eq(logit_of_clamped_one).double()
        neg_inds_0 = logit_Phi0.lt(logit_of_clamped_one).double()

        relevant_pixels = h0.ne(-50).double()

        num_pos_0 = pos_inds_0.float().sum()
        bump =  _bump_shaped_like_logit(logit_Phi0)

        neg_weights_0 = torch.pow(1 - bump, self.beta)

        loss = 0.0

        pos_loss_0_focal_term = torch.pow(1 - torch.sigmoid(h0), self.alpha) * pos_inds_0
        neg_loss_0_focal_term  = torch.pow(torch.sigmoid(h0), self.alpha) * neg_weights_0 * neg_inds_0


        term0 = logit_Phi0 - h0

        term0 = term0 * term0 * relevant_pixels

        pos_loss_0 = (pos_loss_0_focal_term * term0).sum()
        neg_loss_0 = (neg_loss_0_focal_term * term0).sum()


        # with open('verbose_hardcore.pickle','wb') as f:
        #     import pickle
        #     obj = (fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6))
        #     pickle.dump(obj, f)
        # plt.close()           


        if num_pos_0 == 0:
            loss += 0.5*neg_loss_0
        else:
            loss += 0.5*(pos_loss_0 + neg_loss_0) / num_pos_0
            
        fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(3,3, figsize=(20,20))

        ax0.imshow(torch.sigmoid(logit_Phi0[0][0]).cpu(),cmap='gray')
        ax0.set_title('Sigmoid of ground truth')

        ax1.imshow(logit_Phi0[0][0].cpu(),cmap='gray')
        ax1.set_title('Ground truth')

        ax2.imshow(h0[0][0].detach().cpu(),cmap='gray')
        ax2.set_title('Output of network')

        ax3.imshow(neg_weights_0[0][0].detach().cpu(),cmap='gray')
        ax3.set_title('Penalty reduction for pixels close to positive class')

        ax4.imshow(neg_loss_0_focal_term[0][0].detach().cpu(), cmap='gray')
        ax4.set_title('Reweighting for negative class')

        ax5.imshow(pos_loss_0_focal_term[0][0].detach().cpu(), cmap='gray')
        ax5.set_title('Reweighting for positive class')

        loss_vector = term0
        ax6.imshow(loss_vector[0][0].detach().cpu(), cmap='gray')
        ax6.set_title('Loss before adjustment')

        ax7.imshow((pos_loss_0_focal_term * term0)[0][0].detach().cpu(), cmap='gray')
        ax7.set_title('Loss positive class')

        ax8.imshow((neg_loss_0_focal_term * term0)[0][0].detach().cpu(), cmap='gray')
        ax8.set_title('Loss negative class')
        plt.suptitle('Loss: {}'.format(loss))
        plt.show()


        
        if math.isnan(loss):
            test = 0
        
        loss =  loss/self.sigma2

        return loss     

class TestLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(TestLoss, self).__init__()
        self.alpha = int(alpha)
        self.beta = int(beta)

    def forward(self, h, Phi):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''

        h = _sigmoid(h)

        pos_inds = Phi.eq(1).float()
        neg_inds = Phi.lt(1).float()

        neg_weights = torch.pow(1 - Phi, self.beta)

        loss = 0.0

        pos_loss = torch.log(h) * torch.pow(1 - h, self.alpha) * pos_inds
        neg_loss = torch.log(1 - h) * torch.pow(h, self.alpha) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos 
            
        return loss

def _pixel_wise_focal_loss(pred, gt, alpha, beta):

        pred = _sigmoid(pred)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        num_pos  = pos_inds.float().sum()

        neg_weights = torch.pow(1 - gt, beta)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds


        return pos_loss, neg_loss, num_pos

def _sigmoid(x):
    eps = 1e-4
    y = torch.clamp(torch.sigmoid(x), min=eps, max=1-eps)
    return y

