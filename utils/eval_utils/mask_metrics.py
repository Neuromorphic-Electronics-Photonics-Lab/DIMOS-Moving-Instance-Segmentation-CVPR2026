import torch
import numpy as np


def eval_mae_single(pred, gt):
    return torch.abs(pred - gt).mean()


def eval_e_single(y_pred, y, num, cuda=True):
    if cuda:
        score = torch.zeros(num, device=torch.cuda.current_device())
        thlist = torch.linspace(0, 1 - 1e-10, num, device=torch.cuda.current_device())
    else:
        score = torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    y_mean, y_numel = y.mean(), y.numel()
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y_mean
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y_numel - 1 + 1e-20)
    return score.mean()


def S_object(pred, gt):
    fg = torch.where(gt==0, torch.zeros_like(pred), pred)
    bg = torch.where(gt==1, torch.zeros_like(pred), 1 - pred)
    o_fg = sobject(fg, gt)
    o_bg = sobject(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    if torch.isnan(Q):
        print('S_object nan')
    return Q


def sobject(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score


def S_region(pred, gt):
    X, Y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divideGT(gt, X, Y)
    p1, p2, p3, p4 = dividePrediction(pred, X, Y)
    Q1 = ssim(p1, gt1) if w1 != 0 else 0
    Q2 = ssim(p2, gt2) if w1 != 0 else 0
    Q3 = ssim(p3, gt3) if w1 != 0 else 0
    Q4 = ssim(p4, gt4) if w1 != 0 else 0
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q


def centroid(gt, cuda=True):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        if cuda:
            X = torch.eye(1, device=torch.cuda.current_device()) * round(cols / 2)
            Y = torch.eye(1, device=torch.cuda.current_device()) * round(rows / 2)
        else:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        if cuda:
            i = torch.arange(start=0, end=cols, device=torch.cuda.current_device(), dtype=torch.float32)
            j = torch.arange(start=0, end=rows, device=torch.cuda.current_device(), dtype=torch.float32)
        else:
            i = torch.arange(start=0, end=cols, dtype=torch.float32)
            j = torch.arange(start=0, end=rows, dtype=torch.float32)
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)
    return X.long(), Y.long()


def divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def dividePrediction( pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


def ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q


def eval_s_single(pred, gt):
    alpha = 0.5
    y = gt.mean()
    if y == 0:
        x = pred.mean()
        Q = 1.0 - x
    elif y == 1:
        x = pred.mean()
        Q = x
    else:
        gt[gt >= 0.5] = 1
        gt[gt < 0.5] = 0
        Q = alpha * S_object(pred, gt) + (1 - alpha) * S_region(pred, gt)
        if Q.item() < 0:
            Q = torch.FloatTensor([0.0])
    return Q


def eval_f_single(pred, gt):
    def eval_pr(y_pred, y, num):
        prec, recall = torch.zeros(num, device=torch.cuda.current_device()), torch.zeros(num, device=torch.cuda.current_device())
        thlist = torch.linspace(0, 1 - 1e-10, num, device=torch.cuda.current_device())
        y_sum = y.sum()
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y_sum + 1e-20)
        return prec, recall

    beta2 = 0.3
    prec, recall = eval_pr(pred, gt, 255)
    f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
    f_score[f_score != f_score] = 0
    return f_score.mean()


def eval_single_img(pred, gt, is_cuda=True):
    mae = eval_mae_single(pred, gt)
    f = eval_f_single(pred, gt)
    e = eval_e_single(pred, gt, num=255, cuda=is_cuda)
    s = eval_s_single(pred, gt)

    return [mae, f, s, e]


def mask_eval(outputs, inputs):

    if 'pred_mos2' in outputs.keys(): # for aot model
        gt_mos = inputs['mov_segs'][:, -1].unsqueeze(1)
        gt_mask = (gt_mos != 0).float()
        pred_mask = outputs['pred_mask2']
    else:
        if 'ref_img' in inputs.keys():
            gt_mask = (inputs['ref_label'] != 0).float()
        elif 'current_img' in inputs.keys():
            gt_mask = (inputs['current_label'] != 0).float()
        elif 'images' in inputs.keys():
            gt_mask = (inputs['mov_segs'] != 0).float()
        else:
            raise NotImplementedError

        pred_mask = outputs['pred_mask']

    if torch.is_tensor(pred_mask):
        is_cuda = pred_mask.is_cuda
    else:
        pred_mask = torch.tensor(pred_mask)
        gt_mask = torch.tensor(gt_mask)
        is_cuda = False

    if len(pred_mask.shape) == 3:
        pred_mask = pred_mask.unsqueeze(1)

    assert len(pred_mask.shape) == 4 # B, seq_len, H, W
    assert len(gt_mask.shape) == 4 # B, seq_len, H, W

    results = {}

    for batch_idx in range(pred_mask.shape[0]):
        for seq_idx in range(pred_mask.shape[1]):
            pred_single = pred_mask[batch_idx][seq_idx]
            gt_single = gt_mask[batch_idx][seq_idx]
            mae, f, s, e = eval_single_img(pred_single, gt_single, is_cuda)
            if not 'mask_mae_{}'.format(seq_idx) in results.keys():
                results['mask_mae_{}'.format(seq_idx)] = []
            results['mask_mae_{}'.format(seq_idx)].append(mae)

    return results
