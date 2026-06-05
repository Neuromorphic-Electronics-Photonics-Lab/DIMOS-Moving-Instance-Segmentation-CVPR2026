from __future__ import division
import os
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"

import time
import yaml
import math
import random
import argparse
import shutil
import logging
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from omegaconf import DictConfig, ListConfig, OmegaConf
import func_timeout

import torch
# torch.set_num_threads(1)
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler

from eval import validate_once
from utils.seed import set_seed
from utils.ops_utils import copy_to_device
from utils.factory_utils import dataset_factory, model_factory, optimizer_factory, metric_factory
from utils.log_utills import init_logging, format_summary, Boarder


class Trainer:
    def __init__(self, device, cfgs):
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        self.cfgs = cfgs

        self.curr_epoch, self.curr_step = 1, 1
        self.max_mode = self.cfgs.training.max_mode
        if self.max_mode == 'epoch':
            self.max_epoches = self.cfgs.training.max_epoches
            self.valid_every_epoches = self.cfgs.training.valid_every_epoches
        else:
            self.max_steps = self.cfgs.training.max_steps
            self.valid_every_steps = self.cfgs.training.valid_every_steps

        self.device = device
        self.n_gpus = torch.cuda.device_count()
        self.is_main = device is None or device.index == 0
        self.best_metrics = None

        init_logging(os.path.join(self.cfgs.log.full_path, 'train.log'))
        if self.is_main:
            logging.info('Training configurations from {}:\n'.format(cfgs.config) \
                         + OmegaConf.to_yaml(self.cfgs))
            logging.info('Logs will be saved to %s' % self.cfgs.log.full_path)
        else:
            # To show the GPUs together
            time.sleep(0.5)

        if device is None:
            logging.info('No CUDA device detected, using CPU for training')
        else:
            properties = torch.cuda.get_device_properties(device)
            logging.info('Using GPU %d/%d: %s with memory %s GB.' % (device.index + 1, self.n_gpus, \
                properties.name, math.ceil(properties.total_memory / (1024**3))))

            if self.n_gpus >= 1:
                init_process_group('nccl', 'tcp://localhost:{}'.format(cfgs.port), \
                                   world_size=self.n_gpus, rank=self.device.index)
                self.cfgs.model.batch_size = int(self.cfgs.model.batch_size / self.n_gpus)
            cudnn.benchmark = True
            torch.cuda.set_device(self.device)

        # To show the GPUs info together
        time.sleep(1)

        self.boarder = Boarder(self.cfgs.log)
        if self.is_main:
            self.boarder.start()
        else:
            logging.root.disabled = True

        logging.info('Loading training set: %s' % self.cfgs.trainset)
        self.train_dataset = dataset_factory(self.cfgs.trainset)
        self.train_sampler = DistributedSampler(self.train_dataset) if self.n_gpus > 1 else None
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.cfgs.trainset.n_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=self.cfgs.trainset.drop_last,
            collate_fn=mousesis_collate_fn,
        )
        logging.info('Dataset / batch size / iter_per_epoch %d/%d/%d' % ( \
            len(self.train_dataset), self.cfgs.model.batch_size, \
                len(self.train_dataset) // self.cfgs.model.batch_size))

        # TODO: validation on two datasets
        if hasattr(self.cfgs, 'valset'):
            logging.info('Loading validation set: %s' % self.cfgs.valset)
            self.val_dataset = dataset_factory(self.cfgs.valset)
            self.val_sampler = DistributedSampler(self.val_dataset) if self.n_gpus > 1 else None
            self.val_loader = torch.utils.data.DataLoader(
                dataset=self.val_dataset,
                batch_size=self.cfgs.model.batch_size \
                    if not hasattr(self.cfgs.model, 'test_batch_size') else \
                        self.cfgs.model.test_batch_size,
                shuffle=False,
                num_workers=self.cfgs.valset.n_workers,
                pin_memory=True,
                sampler=self.val_sampler,
                collate_fn=mousesis_collate_fn,
            )
            logging.info('Total length: %d' % len(self.val_dataset))
        else:
            logging.info('No validation set, skip...')

        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model).to(device=self.device)
        logging.info('With trainable/total parameters: %d/%d' % (
                        sum([p.numel() for p in self.model.parameters() if p.requires_grad]), \
                        sum([p.numel() for p in self.model.parameters()])
                    ))
        # logging.info('Model configurations:\n' + OmegaConf.to_yaml(self.cfgs.model))

        if self.n_gpus > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            # self.ddp = DistributedDataParallel(self.model, [self.device.index])
            self.ddp = DistributedDataParallel(self.model, [self.device.index])
        else:
            self.ddp = self.model

        if self.cfgs.ckpt.path is not None:
            self.load_ckpt(self.cfgs.ckpt.path, resume=self.cfgs.ckpt.resume)

        logging.info('Creating optimizer: %s' % self.cfgs.training.optimizer)
        logging.info('Creating lr scheduler: %s' % self.cfgs.training.lr.scheduler)
        texture_D_params = list(self.model.texture_discriminator.parameters())
        motion_D_params = list(self.model.motion_discriminator.parameters())
        self.optimizer, self.lr_scheduler, self.lr_mode = optimizer_factory(\
            self.cfgs.training
            , [(n, p) for n, p in self.model.named_parameters() if not ('texture_discriminator' in n or 'motion_discriminator' in n )] # or 'texture_domain_mapper' in n or 'motion_domain_mapper' in n
            , last_epoch=self.curr_epoch - 1, \
                last_step=self.curr_step - 1, train_loader_length=len(self.train_loader),
                # texture_D_params=texture_D_params, motion_D_params=motion_D_params
                )
        # create separate Adam optimizers for two discriminators
        self.opt_texture_D = torch.optim.Adam(texture_D_params, lr=self.cfgs.training.lr.da_lr, weight_decay=0.0)
        self.opt_motion_D = torch.optim.Adam(motion_D_params, lr=self.cfgs.training.lr.da_lr, weight_decay=0.0)
        
        self.amp_scaler = torch.cuda.amp.GradScaler()
        # self.amp_scaler = torch.amp.GradScaler(device='cuda')

        logging.info('Creating metrics for type {} with {} metrics'.format( \
            self.cfgs.metric.type, self.cfgs.metric.names))
        self.metric_funcs = metric_factory(self.cfgs.metric)

    def run(self):
        if self.max_mode == 'epoch':
            logging.info('Start training from {} epoches to {} epoches'.format(self.curr_epoch, self.max_epoches))
        else:
            logging.info('Start training from step {} to {} steps'.format(self.curr_step, self.max_steps))

        while not self.stop_condition():
            if hasattr(self, 'train_sampler') and self.train_sampler is not None:
                self.train_sampler.set_epoch(self.curr_epoch)
            if hasattr(self, 'val_sampler') and self.val_sampler is not None:
                self.val_sampler.set_epoch(self.curr_epoch)

            self.train_one_epoch()

    def stop_condition(self):
        if self.max_mode == 'epoch':
            return self.curr_epoch > self.max_epoches
        else:
            return self.curr_step > self.max_steps

    def train_one_epoch(self):
        self.ddp.train()

        if self.max_mode == 'epoch':
            self.boarder.write_scalar_dict(self.curr_epoch, {'learning_rate': self.optimizer.param_groups[0]['lr']})

        for step, inputs in enumerate(self.train_loader):

            if self.max_mode == 'step' and self.curr_step % self.cfgs.log.save_summary_every_steps == 0:
                self.boarder.write_scalar_dict(self.curr_step, {'learning_rate': self.optimizer.param_groups[0]['lr']})

            inputs = copy_to_device(inputs, self.device)

            
            with torch.amp.autocast('cuda',enabled=self.cfgs.amp):
                self.ddp.forward(inputs, is_Train=True)
                loss = self.model.get_loss()
            
            self.optimizer.zero_grad()
            self.opt_texture_D.zero_grad()
            self.opt_motion_D.zero_grad()
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer)
            if step % self.cfgs.model.optimize_D_per_step == 0:
                self.amp_scaler.step(self.opt_texture_D)
                self.amp_scaler.step(self.opt_motion_D)
            
            
            scale = self.amp_scaler.get_scale()
            self.amp_scaler.update()
            skip_lr_sched = (scale > self.amp_scaler.get_scale())

            self.boarder.push(self.model.get_scalar_summary())

            if hasattr(self.cfgs.log, 'show_summary_every_steps') and \
                self.curr_step % self.cfgs.log.show_summary_every_steps == 0:

                if self.max_mode == 'epoch':
                    info = 'TE: [%d/%d] ' % (self.curr_epoch, self.max_epochs) + \
                        'S: [%d/%d] ' % (step + 1, len(self.train_loader))
                else:
                    info = 'TS(S): [%d(%d)/%d(%d)] ' % (self.curr_step, step + 1, \
                                                        self.max_steps, len(self.train_loader))
                    info += 'E: [%d/%d] ' % (self.curr_epoch, math.ceil(self.max_steps / len(self.train_loader)))

                logging.info(info + '| %s' % (self.boarder.get_step_string()))
                self.print_gpustat()

            if hasattr(self.cfgs.log, 'save_summary_every_steps') and \
                self.curr_step % self.cfgs.log.save_summary_every_steps == 0:
                self.boarder.write_summary_board(self.curr_step, 'train')
                if hasattr(self.cfgs.log, 'save_imagesummary_every_steps') and \
                    self.curr_step % self.cfgs.log.save_imagesummary_every_steps == 0:
                    image_dict = self.model.get_image_summary()
                    self.boarder.write_image_dict(self.curr_step, image_dict, group='viz')
                elif not hasattr(self.cfgs.log, 'save_imagesummary_every_steps'):
                    image_dict = self.model.get_image_summary()
                    self.boarder.write_image_dict(self.curr_step, image_dict, group='viz')

            if self.lr_mode == 'step' and not skip_lr_sched:
                self.lr_scheduler.step()
            if self.max_mode == 'step':
                if isinstance(self.cfgs.log.save_ckpt_every_steps, ListConfig):
                    self.save_ckpt_every_steps = self.cfgs.log.save_ckpt_every_steps[0] \
                        if self.cfgs.log.save_ckpt_every_steps[-1] >= self.curr_step else self.cfgs.log.save_ckpt_every_steps[1]
                else:
                    self.save_ckpt_every_steps = self.cfgs.log.save_ckpt_every_steps

                if isinstance(self.cfgs.training.valid_every_steps, ListConfig):
                    self.valid_every_steps = self.cfgs.training.valid_every_steps[0] \
                        if self.cfgs.training.valid_every_steps[-1] >= self.curr_step else self.cfgs.training.valid_every_steps[1]
                else:
                    self.valid_every_steps = self.cfgs.training.valid_every_steps

                if self.curr_step % self.save_ckpt_every_steps == 0:
                    self.save_ckpt(max_mode='step')
                if self.curr_step % self.valid_every_steps == 0:
                    val_summary = self.validate()
                    self.boarder.write_scalar_dict(self.curr_step, val_summary, group='val')

            self.curr_step += 1

            if self.stop_condition():
                break

        if self.lr_mode == 'epoch':
            self.lr_scheduler.step()
        if self.max_mode == 'epoch':
            if isinstance(self.cfgs.log.save_ckpt_every_epoches, ListConfig):
                self.save_ckpt_every_epoches = self.cfgs.log.save_ckpt_every_epoches[0] \
                    if self.cfgs.log.save_ckpt_every_epoches[-1] <= self.curr_epoch else self.cfgs.log.save_ckpt_every_epoches[1]
            else:
                self.save_ckpt_every_epoches = self.cfgs.log.save_ckpt_every_epoches

            if isinstance(self.cfgs.training.valid_every_epoches, ListConfig):
                self.valid_every_epoches = self.cfgs.training.valid_every_epoches[0] \
                    if self.cfgs.training.valid_every_epoches[-1] <= self.curr_step else self.cfgs.training.valid_every_epoches[1]
            else:
                self.valid_every_epoches = self.cfgs.training.valid_every_epoches

            if self.curr_epoch % self.save_ckpt_every_epoches == 0:
                self.save_ckpt(max_mode='epoch')
            if self.curr_epoch % self.valid_every_epoches == 0:
                val_summary = self.validate()
                self.boarder.write_scalar_dict(self.curr_epoch, val_summary, group='val')

        self.curr_epoch += 1
        # torch.cuda.empty_cache()

    @torch.no_grad()
    def validate(self):
        self.ddp.eval()

        if hasattr(self.cfgs.log, 'show_summary_every_steps'):
            show_steps = self.cfgs.log.show_summary_every_steps
        else:
            show_steps = 20

        if self.max_mode == 'epoch':
            logging.info('Start validation on %s, at epoch %d, for every %d epoches.' % (\
                self.cfgs.valset.name, self.curr_epoch, self.valid_every_epoches))
        else:
            logging.info('Start validation on %s, at step %d, for every %d steps.' % (\
                self.cfgs.valset.name, self.curr_step, self.valid_every_steps))

        val_summary, metrics_summary = validate_once(self.ddp, self.val_loader, self.metric_funcs, \
            self.device, self.n_gpus, self.cfgs.amp, log_path=self.cfgs.log.full_path, \
                show_summary_every_steps=show_steps)
        logging.info('Statistics on validation set: %s' %format_summary(val_summary))

        # torch.cuda.empty_cache()
        self.ddp.train()
        return metrics_summary

    def print_gpustat(self):
        pass

    def save_ckpt(self, filename=None, max_mode=None):
        if max_mode is None:
            max_mode = self.max_mode

        if self.is_main and self.cfgs.log.save_ckpt:
            ckpt_dir = os.path.join(self.cfgs.log.full_path, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            if filename is None:
                filename = 'epoch-%03d.pt' % self.curr_epoch \
                    if max_mode == 'epoch' else 'step-%03d.pt' % self.curr_step
            filepath = os.path.join(ckpt_dir, filename)

            logging.info('Saving checkpoint to %s' % filepath)
            torch.save({
                'last_epoch': self.curr_epoch,
                'last_step': self.curr_step,
                'state_dict': self.model.state_dict(),
                'best_metrics': self.best_metrics
            }, filepath)

    def load_ckpt(self, filepath, resume=True):
        logging.info('Loading checkpoint from %s' % filepath)
        checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
        if resume:
            self.curr_epoch = checkpoint['last_epoch'] + 1
            self.curr_step = checkpoint['last_step'] + 1
            self.best_metrics = checkpoint['best_metrics']
            logging.info('Current best metrics: %s' % str(self.best_metrics))
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)


def create_trainer(device_id, cfgs):
    device = torch.device('cpu' if device_id is None else 'cuda:%d' % device_id)
    trainer = Trainer(device, cfgs)
    trainer.run()

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
    parser.add_argument('--resume', required=False, action='store_true',
                        help='Resume unfinished training')
    parser.add_argument('--port', required=False, type=str, default="",
                        help='DDP port')
    parser.add_argument('--run_name', required=False, type=str, default="",
                        help='log run name')
    args = parser.parse_args()

    # load config
    cfgs = DictConfig(yaml.load(open(args.config, encoding='utf-8'), Loader=yaml.FullLoader))
    if hasattr(cfgs, '_base'):
        base_cfgs = os.path.join(os.path.dirname(args.config), cfgs._base)
        base_cfgs = DictConfig(yaml.load(open(base_cfgs, encoding='utf-8'), Loader=yaml.FullLoader))
        cfgs = OmegaConf.merge(base_cfgs, cfgs)
    cfgs.config = args.config
    cfgs.ckpt.path = args.weights
    cfgs.ckpt.resume = args.resume
    if args.port != "":
        cfgs.port = args.port
    else:
        cfgs.port = "123{}{}".format(random.randint(0, 9), random.randint(0, 9))
    if args.run_name != "":
        cfgs.log.run_name = args.run_name

    if hasattr(cfgs.training, 'max_epoches'):
        cfgs.training.max_mode = 'epoch'
        assert hasattr(cfgs.training, 'valid_every_epoches')
    else:
        cfgs.training.max_mode = 'step'
        assert hasattr(cfgs.training, 'valid_every_steps')

    # create log dir
    if not hasattr(cfgs.log, 'run_name') or cfgs.log.run_name == '' or cfgs.log.run_name == None:
        cfgs.log.run_name = cfgs.model.name + '_' + os.path.basename(args.config).split('.')[0] + '_bs' + str(cfgs.model.batch_size)
        if cfgs.training.max_mode == 'epoch':
            cfgs.log.run_name += '_e' + str(cfgs.training.max_epoches)
        else:
            cfgs.log.run_name += '_i' + str(cfgs.training.max_steps // 10000) + 'w'

    cfgs.log.full_path = os.path.join(cfgs.log.dir, cfgs.log.run_name)
    if os.path.exists(cfgs.log.full_path) and not cfgs.ckpt.resume:

        @func_timeout.func_set_timeout(5)
        def Input_task():
            print('Run "%s" already exists, overwrite it or rename it? [yes/Rename/no]' % cfgs.log.run_name)
            print('waiting 5 seconds to default option: Rename')
            return input()

        try:
            key = Input_task()
        except func_timeout.exceptions.FunctionTimedOut as e:
            print('Timeout! default option: Rename...')
            key = 'R'

        if len(key) == 0 or key[0] == 'N' or key[0] == 'n':
            print('input No, exit...')
            exit(0)
        elif key[0] == 'R' or key[0] == 'r':
            timeStruct = time.localtime(os.path.getatime(cfgs.log.full_path))
            new_name = cfgs.log.full_path + time.strftime('_%Y%m%d_%H%M%S', timeStruct)
            shutil.move(cfgs.log.full_path, new_name)
            print('Rename old folder to %s and continue with %s...' % (new_name, cfgs.log.full_path))
        elif key[0] == 'Y' or key[0] == 'y' or key[0] == '1':
            shutil.rmtree(cfgs.log.full_path, ignore_errors=True)
            print('Delete old folder %s and overwrite continue...' % (cfgs.log.full_path))
        else:
            print('Unknow command, exit...')
            exit(0)

    os.makedirs(cfgs.log.full_path, exist_ok=True)

    set_seed(1234)

    # create trainers
    if torch.cuda.device_count() == 0:  # CPU
        create_trainer(None, cfgs)
    elif torch.cuda.device_count() >= 1:  # GPUs
        mp.spawn(create_trainer, (cfgs,), torch.cuda.device_count())
