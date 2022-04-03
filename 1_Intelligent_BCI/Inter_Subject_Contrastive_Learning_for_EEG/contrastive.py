import torch


def inter_subject_contrastive_loss(feats, labels, subject_ids, tau):
    N = feats.size(0)

    labels = labels.contiguous().view(-1, 1)
    cls_mask = torch.eq(labels, labels.T).float().cuda()

    subject_ids = subject_ids.contiguous().view(-1, 1)
    subject_mask = torch.eq(subject_ids, subject_ids.T).float().cuda()

    # the default mask includes all samples but the anchor itself
    default_mask = torch.ones((N, N)).fill_diagonal_(0).cuda()

    # construct the positive set
    positive_set_mask = default_mask * cls_mask * (1 - (cls_mask * subject_mask))
    
    # construct the anchor set
    anchor_set_mask = default_mask * (1 - ((1 - cls_mask) * (1 - subject_mask))) * (1 - (cls_mask * subject_mask))

    feats = feats / torch.norm(feats, p=2, dim=1).unsqueeze(1)

    sim_matrix = torch.matmul(feats, torch.transpose(feats, 0, 1)) / tau

    sim_matrix_exp = torch.exp(sim_matrix)
    sim_matrix_exp = sim_matrix_exp.clone().fill_diagonal_(0)

    scores = (sim_matrix_exp * positive_set_mask).sum(dim=0) / (sim_matrix_exp * anchor_set_mask).sum(dim=0)

    loss_contrast = -torch.log(scores).mean()

    return loss_contrast