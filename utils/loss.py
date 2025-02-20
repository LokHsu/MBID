import torch
import numpy as np
import torch.nn.functional as F

def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_score = torch.sum(user_emb * pos_emb, dim=1)
    neg_score = torch.sum(user_emb * neg_emb, dim=1)
    return torch.sum(F.softplus(neg_score - pos_score))


def reg_loss(user_emb, pos_emb, neg_emb):
    reg_loss = 0.5 * (user_emb.norm(2).pow(2) +
                      pos_emb.norm(2).pow(2) +
                      neg_emb.norm(2).pow(2))
    return reg_loss


def infonce_loss(tar_user_emb, aux_user_emb, tau=0.5):
    x_norm = F.normalize(tar_user_emb)
    y_norm = F.normalize(aux_user_emb)
    
    pos_score = torch.sum(x_norm * y_norm, dim=1)
    pos_score = torch.exp(pos_score / tau)

    neg_score = torch.matmul(x_norm, y_norm.T)
    neg_score = torch.sum(torch.exp(neg_score / tau), dim=1)

    cl_loss = -torch.sum(torch.log(pos_score / neg_score))
    return cl_loss


def hsic_loss(x, y, sigma, unbiased=True):
    k_x = kernel_matrix(x, sigma)
    k_y = kernel_matrix(y, sigma)

    m = x.shape[0]

    if unbiased:
        '''
        PyTorch Implementation of Unbiased Estimator of HSIC from:
        Feature Selection via Dependence Maximization
        by Song Le, et al.
        '''
        # k_x_hat = k_x - torch.diag(k_x)
        # k_y_hat = k_y - torch.diag(k_y)
        k_x = kernel_matrix(x, sigma)
        k_y = kernel_matrix(y, sigma)
        
        k_xy = torch.mm(k_x, k_y)
        h = (torch.trace(k_xy)
            + torch.mean(k_x) * torch.mean(k_y)
            - 2 * torch.sum(k_xy) / m)
        hsic_loss =  h / (m * (m - 1))
    else:
        '''
        PyTorch Implementation of Biased Estimator of HSIC from:
        Measuring Statistical Dependence with Hilbert-Schmidt Norms
        by Gretton Arthur, et al.
        '''
        kh = k_x - k_x.mean(0, keepdim=True)
        lh = k_y - k_y.mean(0, keepdim=True)
        hsic_loss = torch.trace(kh @ lh / (m - 1) ** 2)
    return hsic_loss


def kernel_matrix(x, sigma):
    '''RBF kernel'''
    return torch.exp((x @ x.t() - 1) / sigma)


def calc_bpr_loss(user, item_i, item_j, user_embs, item_embs, behaviors, device):
    bpr_loss_list = [None] * len(behaviors)
    for i in range(len(behaviors)):
        act_user_idx = np.where(item_i[i].cpu().numpy() != -1)[0]

        act_user = user[act_user_idx].long().to(device)
        act_user_pos = item_i[i][act_user_idx].long().to(device)
        act_user_neg = item_j[i][act_user_idx].long().to(device)

        act_user_emb = user_embs[i][act_user]
        act_user_pos_emb = item_embs[i][act_user_pos]
        act_user_neg_emb = item_embs[i][act_user_neg]

        bpr_loss_list[i] = bpr_loss(act_user_emb, act_user_pos_emb, act_user_neg_emb)

        if i == len(behaviors) - 1:
            l2_loss = reg_loss(act_user_emb, act_user_pos_emb, act_user_neg_emb)

    return sum(bpr_loss_list) / len(bpr_loss_list), l2_loss


def calc_infonce_loss(tar_user_emb, gci_embs, batch_users, behaviors):
    cl_loss_list = [None] * len(behaviors)
    tar_user_emb = tar_user_emb[batch_users]
    for i in range(len(behaviors) - 1):
        gci_emb = gci_embs[i][batch_users]
        cl_loss_list[i] = infonce_loss(tar_user_emb, gci_emb)

    return sum(cl_loss_list) / len(cl_loss_list)


def calc_hsic_loss(tar_user_emb, sci_embs, batch_users, behaviors, sigma):
    hsic_loss_list = [None] * len(behaviors)
    tar_user_emb = tar_user_emb[batch_users]
    for i in range(len(behaviors) - 1):
        sci_emb = sci_embs[i][batch_users]
        hsic_loss_list[i] = hsic_loss(tar_user_emb, sci_emb, sigma)

    return sum(hsic_loss_list) / len(hsic_loss_list)
