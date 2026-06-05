import torch
from .upflow_tools import tools, network_tools, loss_functions


def calculate(im1, im2, flow_pred):
    mask = torch.ones_like(im1[:, 0:1])
    smooth_loss = network_tools.edge_aware_smoothness_order1(img=im1, pred=flow_pred)
    im1_warp = tools.boundary_dilated_warp.warp_im(im2, flow_pred).float()
    photo_loss = network_tools.photo_loss_multi_type(im1, im1_warp, photo_loss_type='abs_robust', \
                                                     photo_loss_delta=0.4, photo_loss_use_occ=False)
    census_loss = loss_functions.census_loss_torch(im1, im1_warp, mask=mask, q=0.4, charbonnier_or_abs_robust=False, if_use_occ=False, averge=True)

    return smooth_loss, photo_loss, census_loss, im1_warp


def flow_unsup_loss(flow_preds, image1, image2, gamma=0.8):

    if isinstance(flow_preds, torch.Tensor):
        flow_preds = [flow_preds]

    n_predictions = len(flow_preds)    
    smooth_loss = 0.0
    photo_loss = 0.0
    census_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        isloss, iploss, icloss, im1_warp = calculate(image1, image2, flow_preds[i])
        smooth_loss += (i_weight * isloss).mean()
        photo_loss += (i_weight * iploss).mean()
        census_loss += (i_weight * icloss).mean()

    return smooth_loss, photo_loss, census_loss, im1_warp
