from __future__ import division
import os
os.environ["KMP_BLOCKTIME"] = "0"
import time
import math
import yaml
import random
import argparse
import numpy as np
import logging
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from omegaconf import DictConfig, OmegaConf

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.append('.')
from utils.seed import set_seed
from utils.ops_utils import copy_to_device, dist_reduce_sum
from utils.factory_utils import dataset_factory, model_factory, metric_factory
from utils.log_utills import format_summary


@torch.no_grad()
def validate_once(model, val_loader, metric_funcs, device, n_gpus, amp=False, log_path='./', show_summary_every_steps=20):
    model.eval()
    metrics_summary = {}
    start_time = time.time()
    each_times = []
    for step, inputs in enumerate(val_loader):
        inputs = copy_to_device(inputs, device)

        torch.cuda.synchronize()
        tm = time.time()

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model.forward(inputs, is_Train=False)

        torch.cuda.synchronize()
        elapsed = time.time() - tm
        each_times.append(elapsed)

        batch_summary = {}
        for metric in metric_funcs:
            batch_summary.update(metric(outputs, inputs))

        for key in batch_summary.keys():
            if not key in metrics_summary.keys():
                metrics_summary[key] = batch_summary[key]
            elif isinstance(batch_summary[key], list):
                metrics_summary[key].extend(batch_summary[key])
            elif isinstance(batch_summary[key], torch.Tensor):
                metrics_summary[key] = torch.concat((metrics_summary[key], batch_summary[key]), dim=0)
            else:
                raise NotImplementedError

        if (step + 1) % show_summary_every_steps == 0 or (step + 1) == len(val_loader):

            info_str = "Val steps: [%d/%d] | " % (step + 1, len(val_loader))
            for idx, key in enumerate(metrics_summary.keys()):
                if isinstance(metrics_summary[key], list):
                    info_str += "{}:{:8.6f}, ".format(key, sum(metrics_summary[key]) / len(metrics_summary[key]))
                elif isinstance(metrics_summary[key], torch.Tensor):
                    info_str += "{}:{:8.6f}, ".format(key, torch.sum(metrics_summary[key]) / len(metrics_summary[key]))
                else:
                    raise NotImplementedError
            info_str += "time: {:5.2f}s.".format(time.time() - start_time)
            logging.info(info_str)

    result_summary = {}
    for key in metrics_summary.keys():
        if isinstance(metrics_summary[key], list): # except flow and coco eval
            result_summary[key] = dist_reduce_sum(sum(metrics_summary[key]), n_gpus) / len(val_loader.dataset)

    for metric in metric_funcs: # for flow and coco eval
        if hasattr(metric, 'summary'):  # for coco eval
            result_summary.update(metric.summary(
                temp_dir=log_path, gpu_idx=device.index, n_gpu=n_gpus))
            dist_reduce_sum(0., n_gpus)  # for sync
            if hasattr(metric, 'reset'):
                metric.reset(device.index)
        if hasattr(metric, 'flow_outlier'):
            result_summary.update(metric.flow_outlier(metrics_summary['epe']))

    val_summary = {}
    val_summary.update(result_summary)
    val_summary.update({
        'total_time': time.time() - start_time,
        'each_time': np.mean(each_times),
        'iters': len(val_loader),
        'data_length': len(val_loader.dataset),
    })
    return val_summary, result_summary


def create_evaluate(device_id, cfgs):
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    device = torch.device(
        'cpu' if device_id is None else 'cuda:%d' % device_id)

    n_gpus = torch.cuda.device_count()
    is_main = device is None or device.index == 0

    logging.root = logging.RootLogger('INFO')

    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)


    if device is None:
        logging.info('No CUDA device detected, using CPU for training')
    else:
        properties = torch.cuda.get_device_properties(device)
        logging.info('Using GPU %d/%d: %s with memory %s GB.' % (device.index + 1, n_gpus,
                                                                 properties.name, math.ceil(properties.total_memory / (1024**3))))

        if n_gpus > 1:
            init_process_group('nccl', 'tcp://localhost:{}'.format(cfgs.port),
                               world_size=n_gpus, rank=device.index)
            cfgs.model.batch_size = int(cfgs.model.batch_size / n_gpus)
        cudnn.benchmark = True
        torch.cuda.set_device(device)

    if not is_main:
        logging.root.disabled = True

    if hasattr(cfgs, 'valset'):
        val_dataset = dataset_factory(cfgs.valset)
        val_sampler = DistributedSampler(val_dataset) if n_gpus > 1 else None
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=cfgs.model.batch_size,
            shuffle=False,
            num_workers=cfgs.valset.n_workers,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=mousesis_collate_fn
        )
        # logging.info('Loading validation set: %s, total length: %d' % (cfgs.valset.name, len(val_dataset)))
    else:
        logging.info('No validation set, exit...')
        exit()

    model = model_factory(cfgs.model, eval_mode=True).to(device=device)
    logging.info('Creating model: %s, with trainable/total parameters: %d/%d' % (
        cfgs.model.name,
        sum([p.numel() for p in model.parameters() if p.requires_grad]),
        sum([p.numel() for p in model.parameters()])
    ))
    # logging.info('Model configurations:\n' + OmegaConf.to_yaml(cfgs.model))

    if n_gpus > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, [device.index])

    if cfgs.ckpt.path is None:
        logging.info('Please input the weight path to evaluate.')
        exit(0)
    else:
        logging.info('Loading checkpoint from %s' % cfgs.ckpt.path)
        state_dict = torch.load(cfgs.ckpt.path, map_location=torch.device("cpu"))
        if "model" in state_dict.keys():
            state_dict = state_dict.pop("model")
        elif 'state_dict' in state_dict.keys():
            state_dict = state_dict.pop("state_dict")
        elif 'model_state_dict' in state_dict.keys():
            state_dict = state_dict.pop("model_state_dict")

        if "module." in list(state_dict.keys())[0]:
            for key in list(state_dict.keys()):
                state_dict.update({key[7:]:state_dict.pop(key)})

        model.load_state_dict(state_dict, strict=False)

    logging.info('Creating metrics for type {} with {} metrics'.format(
        cfgs.metric.type, cfgs.metric.names))
    metric_funcs = metric_factory(cfgs.metric)

    if hasattr(cfgs.log, 'show_summary_every_steps'):
        show_steps = cfgs.log.show_summary_every_steps
    else:
        show_steps = 20

    logging.info('Start validation on %s, total length: %d, iters %d ...' % (
        cfgs.valset.name, len(val_dataset), len(val_loader)))

    val_summary, _ = validate_once(model, val_loader, metric_funcs, device, n_gpus, cfgs.amp,
                                   show_summary_every_steps=show_steps)
    logging.info('Statistics: %s' % format_summary(val_summary))

    torch.cuda.empty_cache()

def mousesis_collate_fn(batch):
    """
    Custom collate function for MouseSIS dataset to handle variable-length gt_bboxes.
    Args:
        batch (list): A list of data_dicts from MouseSIS dataset.
    Returns:
        dict: A batched dictionary with gt_bboxes handled as a list of tensors.
    """
    collated_batch = {}

    # Iterate over keys in the first sample to initialize the collated batch
    for key in batch[0].keys():
        if key == 'gt_bboxes':
            # Handle variable-length gt_bboxes as a list of tensors
            collated_batch[key] = [sample[key] for sample in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            # Stack tensors for other fields
            collated_batch[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], list):
            # Concatenate lists for fields like 'images' or 'event_voxels'
            collated_batch[key] = [item for sample in batch for item in sample[key]]
        else:
            # Directly copy other fields
            collated_batch[key] = [sample[key] for sample in batch]

    return collated_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Path to the configuration (YAML format)')
    parser.add_argument('--weights', required=False, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--port', required=False, type=str, default="",
                        help='DDP port')
    parser.add_argument('--gpus', required=False, type=int, default=1,
                        help='gpus number')
    parser.add_argument('-bs', '--batch_size', required=False, type=int, default=0,
                        help='batch_size')
    parser.add_argument('--amp', required=False, action='store_true',
                        help='amp')
    args = parser.parse_args()

    # load config
    cfgs = DictConfig(yaml.load(open(args.config, encoding='utf-8'), Loader=yaml.FullLoader))
    if hasattr(cfgs, '_base'):
        base_cfgs = os.path.join(os.path.dirname(args.config), cfgs._base)
        base_cfgs = DictConfig(yaml.load(open(base_cfgs, encoding='utf-8'), Loader=yaml.FullLoader))
        cfgs = OmegaConf.merge(base_cfgs, cfgs)
    cfgs.config = args.config
    cfgs.ckpt.path = args.weights
    cfgs.gpus = args.gpus
    cfgs.amp = args.amp
    if args.batch_size >= 1:
        cfgs.model.batch_size = args.batch_size
    if args.port != "":
        cfgs.port = args.port
    else:
        cfgs.port = "123{}{}".format(
            random.randint(0, 9), random.randint(0, 9))

    set_seed(1234)

    # create trainers
    if torch.cuda.device_count() == 0:  # CPU
        create_evaluate(None, cfgs)
    elif torch.cuda.device_count() == 1:  # Single GPU
        create_evaluate(0, cfgs)
    elif torch.cuda.device_count() > 1:  # Multiple GPUs
        mp.spawn(create_evaluate, (cfgs,), torch.cuda.device_count())
