import torch

# exclude extremly large displacements
MAX_FLOW = 400


def flow_supervised_loss(flow_preds, flow_gt, valid=None, gamma=0.8, max_flow=MAX_FLOW):
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    if valid is None:
        valid = (mag < max_flow)
    else:
        valid = (valid >= 0.5) & (mag < max_flow)

    if isinstance(flow_preds, torch.Tensor):
        flow_preds = [flow_preds]

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    return flow_loss


def flow_supervised_loss_list(flow_preds, flow_gts):
    flow_loss = 0.0
    assert len(flow_preds) >= 1

    assert isinstance(flow_preds, list)
    assert isinstance(flow_gts, list)

    for i in range(min(len(flow_preds), len(flow_gts))):
        flow_loss += flow_supervised_loss(flow_preds[i], flow_gts[i])
    return flow_loss
