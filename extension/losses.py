import torch
import torch.nn as nn
from torch import sigmoid, logit
import math 
import matplotlib.pyplot as plt

class TrainLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, sigma2=2):
        super(TrainLoss, self).__init__()
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.sigma2 = int(sigma2)

    def forward(self, h0, h1, logit_Phi0, logit_Phi1):
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

        relevant_pixels_1 = h1.ne(-50).double()

        pos_inds_0 = logit_Phi0.eq(logit_of_clamped_one).double()
        neg_inds_0 = logit_Phi0.lt(logit_of_clamped_one).double()

        pos_inds_1 = logit_Phi1.eq(logit_of_clamped_one).double()
        neg_inds_1 = logit_Phi1.lt(logit_of_clamped_one).double()

        num_pos_0 = pos_inds_0.float().sum()
        num_pos_1 = pos_inds_1.float().sum()

        # bump_0 = _bump_shaped_like_logit(logit_Phi0)
        # bump_1 = _bump_shaped_like_logit(logit_Phi1)
        neg_weights_0 = torch.pow(1 - torch.sigmoid(logit_Phi0), self.beta)
        neg_weights_1 = torch.pow(1 - torch.sigmoid(logit_Phi1), self.beta)

        loss = 0.0

        pos_loss_0_focal_term = torch.pow(1 - _sigmoid(h0), self.alpha) * pos_inds_0
        neg_loss_0_focal_term  = torch.pow(_sigmoid(h0), self.alpha) * neg_weights_0 * neg_inds_0

        pos_loss_1_focal_term  = torch.pow(1 - _sigmoid(h1), self.alpha) * pos_inds_1
        neg_loss_1_focal_term  = torch.pow(_sigmoid(h1), self.alpha) * neg_weights_1 * neg_inds_1

        term0 = logit_Phi0 - h0
        term1 = logit_Phi1 - h1

        term0 = term0 * term0
        term1 = term1 * term1

        pos_loss_0 = (pos_loss_0_focal_term * term0).sum()
        neg_loss_0 = (neg_loss_0_focal_term * term0).sum()

        pos_loss_1 = (pos_loss_1_focal_term * term1 * relevant_pixels_1).sum()
        neg_loss_1 = (neg_loss_1_focal_term * term1 * relevant_pixels_1).sum()



        # with open('verbose_hardcore.pickle','wb') as f:
        #     import pickle
        #     obj = (fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6))
        #     pickle.dump(obj, f)
        # plt.close()           


        if num_pos_0 == 0:
            loss += 0.5*neg_loss_0
        else:
            loss += 0.5*(pos_loss_0 + neg_loss_0) / num_pos_0
            
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


        if num_pos_1 == 0:
            loss += 0.5*neg_loss_1
        else:
            loss += 0.5*(pos_loss_1 + neg_loss_1) / num_pos_1
        
        # if math.isnan(loss):
        #     test = 0
        
        loss =  loss/self.sigma2

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



def _sigmoid(x):
    eps = 1e-4
    y = torch.clamp(torch.sigmoid(x), min=eps, max=1-eps)
    return y



def _bump_shaped_like_logit(x):
    logit_min_value = torch.logit(torch.tensor(1e-16,  dtype=torch.float64))
    logit_max_value = torch.logit(torch.tensor(1-1e-16,  dtype=torch.float64))
    x_lowest_0 =  x - logit_min_value
    return x_lowest_0/(logit_max_value-logit_min_value)


# def _logit(x):
#     y = torch.clamp(logit(x), min=logit(torch.tensor(1e-4)), max=logit(torch.tensor(1-1e-4)))
#     return y