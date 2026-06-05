import torch


def pytorch_iou(pred, target, epsilon=1e-6):
    '''
    pred: [h, w]
    target: [h, w]
    '''
    if len(pred.shape) == 2:
        now_pred = pred.unsqueeze(0)
        now_target = target.unsqueeze(0)
    now_obj_num = now_target.max()

    obj_ids = torch.arange(0, now_obj_num + 1, device=now_pred.device).int().view(-1, 1, 1)
    if obj_ids.size(0) == 1:  # only contain background
        return torch.ones((1), device=pred.device)
    else:
        obj_ids = obj_ids[1:]
        now_pred = (now_pred == obj_ids).float()
        now_target = (now_target == obj_ids).float()

        intersection = (now_pred * now_target).sum((1, 2))
        union = ((now_pred + now_target) > 0).float().sum((1, 2))

        now_iou = (intersection + epsilon) / (union + epsilon)
        return now_iou.mean()


def iou_eval(outputs, inputs):

    if 'pred_mos2' in outputs.keys(): # for aot model
        gt_mos = inputs['mov_segs'][:, -1].unsqueeze(1)
        pred_mos = outputs['pred_mos2']
        pred_mask = outputs['pred_mask2']
    else:
        if 'ref_img' in inputs.keys():
            gt_mos = inputs['ref_label']
        elif 'current_img' in inputs.keys():
            gt_mos = inputs['current_label']
        elif 'images' in inputs.keys():
            gt_mos = inputs['mov_segs']
        else:
            raise NotImplementedError

        pred_mask = outputs['pred_mask']
        pred_mos = outputs['pred_mos']

    gt_mask = (gt_mos != 0).float()
    if not torch.is_tensor(pred_mask):
        gt_mos = torch.tensor(gt_mos)
        gt_mask = torch.tensor(gt_mask)
        pred_mos = torch.tensor(pred_mos)
        pred_mask = torch.tensor(pred_mask)

    if len(pred_mask.shape) == 3:
        pred_mask = pred_mask.unsqueeze(1)

    assert len(pred_mask.shape) == 4 # B, seq_len, H, W
    assert len(gt_mask.shape) == 4 # B, seq_len, H, W

    results = {}

    for batch_idx in range(pred_mask.shape[0]):
        for seq_idx in range(pred_mask.shape[1]):
            pred_single = pred_mask[batch_idx][seq_idx]
            gt_single = gt_mask[batch_idx][seq_idx]
            if torch.count_nonzero(gt_single) != 0:
                mask_iou = pytorch_iou(pred_single, gt_single)
            else:
                mask_iou = torch.tensor(0.)
            if not 'mask_iou_{}'.format(seq_idx) in results.keys():
                results['mask_iou_{}'.format(seq_idx)] = []
            results['mask_iou_{}'.format(seq_idx)].append(mask_iou)

    for batch_idx in range(pred_mos.shape[0]):
        for seq_idx in range(pred_mos.shape[1]):
            pred_single = pred_mos[batch_idx][seq_idx].clone()
            gt_single = gt_mos[batch_idx][seq_idx]

            gt_unique_inst_ids = torch.unique(gt_single[gt_single > 0])
            mos_ious = []
            for gt_inst_id in gt_unique_inst_ids:
                gt_mask = (gt_single == gt_inst_id).float()

                pred_unique_inst_ids = torch.unique(pred_single[pred_single > 0])
                per_ious = [
                    pytorch_iou((pred_single == i).float(), gt_mask) \
                        for i in pred_unique_inst_ids
                ]

                if len(per_ious) > 0:
                    matched_iou = max(per_ious)
                    mos_ious.append(matched_iou)
                    pred_matched_id = pred_unique_inst_ids[per_ious.index(matched_iou)]
                    pred_single[pred_single == pred_matched_id] = 0
                else:
                    mos_ious.append(torch.tensor(0.))
            if len(mos_ious) > 0:
                mos_iou = sum(mos_ious) / len(mos_ious)
            else:
                mos_iou = torch.tensor(0.)
            if not 'mos_iou_{}'.format(seq_idx) in results.keys():
                results['mos_iou_{}'.format(seq_idx)] = []
            results['mos_iou_{}'.format(seq_idx)].append(mos_iou)

    return results
