from typing import Any
import torch

# exclude extremly large displacements
MAX_FLOW = 400


class FlowEval:
    def __init__(self, px1_only=False):
        self.px1_only = px1_only

    def __call__(self, outputs, inputs):
        
        flow_preds = outputs['pred_flow']
        flow_gt = inputs['flow_2d']
        batch_size = flow_gt.shape[0]

        if isinstance(flow_preds, torch.Tensor):
            flow_preds = [flow_preds]

        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (mag < MAX_FLOW)

        epe_all = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()

        epe_list = []
        epe_1px = []
        epe_3px = []
        epe_5px = []
        for b in range(batch_size):
            epe = epe_all[b][valid[b]]
            epe_list.append(epe.mean())
            epe_1px.append((epe < 1.).float().mean() * 100)
            if not self.px1_only:
                epe_3px.append((epe < 3.).float().mean() * 100)
                epe_5px.append((epe < 5.).float().mean() * 100)

        epe_all = epe_all.view(-1)[valid.view(-1)]

        result_dict = {
            'epe' : epe_list,
            'epe_1px': epe_1px,
        }
        if not self.px1_only:
            result_dict.update({
                'epe_3px': epe_3px,
                'epe_5px': epe_5px,
            })

        return result_dict


