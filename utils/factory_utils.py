import torch
from omegaconf import DictConfig

def dataset_factory(cfgs: DictConfig):
    from utils.mos_dataloaders.mos_transform import AugMosData

    if cfgs.name == 'kubric':
        from utils.mos_dataloaders.ekubric import KubricData
        dataset = KubricData(cfgs)
    elif cfgs.name == 'evimov1':
        from utils.mos_dataloaders.evimo import EVIMOv1
        dataset = EVIMOv1(cfgs)
    elif cfgs.name == 'sevd':
        from utils.mos_dataloaders.sevd import SEVD
        dataset = SEVD(cfgs)
    elif cfgs.name == 'mousesis':
        from utils.mos_dataloaders.mousesis import MouseSIS
        dataset = MouseSIS(cfgs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % cfgs.name)

    return AugMosData(cfgs.augmentation, dataset)


def model_factory(cfgs: DictConfig, eval_mode=False):
    if cfgs.name == 'dimos':
        from networks.models.dimos import DIMOS
        return DIMOS(cfgs, eval_mode=eval_mode)
    else:
        raise NotImplementedError('Unknown model name: %s' % cfgs.name)


def optimizer_factory(cfgs, named_params, last_epoch, last_step, train_loader_length, texture_D_params=None, motion_D_params=None):
    param_groups = [
        {'params': [p for name, p in named_params if 'weight' in name],
         'weight_decay': cfgs.weight_decay,
         'lr': cfgs.lr.init_value},  # main model parameters
        {'params': [p for name, p in named_params if 'bias' in name],
         'weight_decay': cfgs.bias_decay,
         'lr': cfgs.lr.init_value},  # main model parameters
    ]

    # discriminator parameters in separate param group with different learning rate
    if texture_D_params is not None:
        param_groups.append({'params': texture_D_params, 'lr': cfgs.lr.da_lr, 'weight_decay': 0.0})
    if motion_D_params is not None:
        param_groups.append({'params': motion_D_params, 'lr': cfgs.lr.da_lr, 'weight_decay': 0.0})

    if cfgs.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params=param_groups,
            eps=1e-7
        )
    elif cfgs.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params=param_groups,
            momentum=cfgs.lr.momentum
        )
    else:
        raise NotImplementedError('Unknown optimizer: %s' % cfgs.optimizer)

    if cfgs.lr.scheduler == 'OneCycleLR':
        lr_mode = 'step'
        if cfgs.max_mode == 'epoch':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, \
                max_lr=cfgs.lr.init_value, steps_per_epoch=train_loader_length, \
                    epochs=cfgs.max_epoches)
        else:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, \
                max_lr=cfgs.lr.init_value, total_steps=cfgs.max_steps)
    else: # StepLR
        if isinstance(cfgs.lr.decay_milestones, int):
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=cfgs.lr.decay_milestones,
                gamma=cfgs.lr.decay_rate
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=cfgs.lr.decay_milestones,
                gamma=cfgs.lr.decay_rate
            )
        lr_mode = 'epoch'

    if cfgs.max_mode == 'epoch':
        for _ in range(last_epoch):
            for i in range(train_loader_length):
                optimizer.step()
                if lr_mode == 'step':
                    lr_scheduler.step()

            if lr_mode == 'epoch':
                lr_scheduler.step()
    else:
        for i in range(last_step):
            optimizer.step()
            lr_scheduler.step()

    return optimizer, lr_scheduler, lr_mode


def metric_factory(cfgs: DictConfig):
    type = cfgs.type
    metric_names = list(cfgs.names)
    metric_funs = []

    def single_metric(name):
        if name.lower() == 'mask':
            from utils.eval_utils.mask_metrics import mask_eval
            return mask_eval
        elif name.lower() == 'iou':
            from utils.eval_utils.iou_metrics import iou_eval
            return iou_eval
        elif name.lower() == 'coco':
            from utils.eval_utils.coco_metrics import CocoEval
            return CocoEval()
        elif name.lower() == 'epe1px':
            from utils.eval_utils.flow_metrics import FlowEval
            return FlowEval(px1_only=True)
        elif name.lower() == 'bbox':
            from utils.eval_utils.bbox_metrics import bbox_eval
            return bbox_eval
        else:
            raise NotImplementedError('{} not implemented'.format(name))

    if isinstance(metric_names, str):
        metric_funs.append(single_metric(metric_names))
    elif isinstance(metric_names, (list)):
        for name in metric_names:
            metric_funs.append(single_metric(name))

    return metric_funs
