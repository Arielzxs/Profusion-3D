import torch
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax(probs, labels, classes='present', per_image=False, ignore=None):
    """
    PyTorch 版 Lovasz-Softmax（精简自 bermanmaxim/LovaszSoftmax）
    probs: [N, C, ...] softmax 后概率
    labels: [N, ...] int64
    """
    if probs.numel() == 0:
        return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = range(C) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if ignore is not None:
            mask = (labels != ignore).float()
            fg = fg * mask
            prob = probs[:, c, ...] * mask
        else:
            prob = probs[:, c, ...]
        errors = (fg - prob).abs().view(-1)
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg.view(-1)[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return torch.tensor(0., device=probs.device)
    return sum(losses) / len(losses)