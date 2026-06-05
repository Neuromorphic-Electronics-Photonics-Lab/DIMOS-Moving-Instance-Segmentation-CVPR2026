import torch

def bbox_iou(pred, target, epsilon=1e-6):
    """
    Calculate IoU (Intersection over Union) between predicted and ground truth bboxes.
    Args:
        pred (Tensor): Predicted bboxes, shape (num_pred, 4).
        target (Tensor): Ground truth bboxes, shape (num_gt, 4).
    Returns:
        Tensor: IoU matrix of shape (num_pred, num_gt).
    """
    pred = pred.unsqueeze(1)  # (num_pred, 1, 4)
    target = target.unsqueeze(0)  # (1, num_gt, 4)

    # Calculate intersection
    inter_x1 = torch.max(pred[..., 0], target[..., 0])
    inter_y1 = torch.max(pred[..., 1], target[..., 1])
    inter_x2 = torch.min(pred[..., 2], target[..., 2])
    inter_y2 = torch.min(pred[..., 3], target[..., 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Calculate union
    pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / (union_area + epsilon)
    return iou

def bbox_eval(outputs, inputs):
    """
    Evaluate bbox predictions using mean IoU (mIoU).
    Args:
        outputs (dict): Model outputs containing 'pred_bboxes'.
        inputs (dict): Ground truth inputs containing 'gt_bboxes'.
    Returns:
        dict: Evaluation results containing mIoU for each sequence.
    """
    pred_bboxes = outputs['pred_bboxes']  # List[Tensor], each Tensor is (num_pred, 4)
    gt_bboxes = inputs['gt_bboxes']  # List[List[Tensor]], each Tensor is (num_gt, 4)

    results = {}

    for batch_idx in range(len(pred_bboxes)):  # iterate over each batch
        pred_bbox = pred_bboxes[batch_idx]
        gt_bbox = gt_bboxes[batch_idx][0].to(pred_bbox.device)  # take the first GT bbox

        # if predicted or GT bboxes are empty or pred is None, return IoU as 0
        if pred_bbox is None or pred_bbox.numel() == 0 or gt_bbox.numel() == 0:
            seq_iou = torch.tensor(0.0)
        else:
            # compute IoU matrix
            iou_matrix = bbox_iou(pred_bbox, gt_bbox)

            # iterate over each GT bbox, find the predicted bbox with max IoU
            gt_matched_ious = []
            gt_matched_pred_ids = []
            for gt_idx in range(iou_matrix.shape[1]):  # iterate over each GT bbox
                max_iou, pred_idx = iou_matrix[:, gt_idx].max(0)  # find predicted bbox with max IoU
                gt_matched_ious.append(max_iou.item())
                gt_matched_pred_ids.append(pred_idx.item())

            # iterate over each predicted bbox, find the GT bbox with max IoU
            pred_matched_ious = []
            pred_matched_gt_ids = []
            for pred_idx in range(iou_matrix.shape[0]):  # iterate over each predicted bbox
                max_iou, gt_idx = iou_matrix[pred_idx, :].max(0)  # find GT bbox with max IoU
                pred_matched_ious.append(max_iou.item())
                pred_matched_gt_ids.append(gt_idx.item())

            # compute mIoU: average of matched IoUs between GT and predicted bboxes
            seq_iou = (sum(gt_matched_ious) + sum(pred_matched_ious)) / (
                len(gt_matched_ious) + len(pred_matched_ious)
            )

        # save IoU for each sequence frame
        if f"bbox_iou_0" not in results:
            results[f"bbox_iou_0"] = []
        results[f"bbox_iou_0"].append(seq_iou)

    return results