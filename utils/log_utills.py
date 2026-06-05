import os
import sys
import time
import logging
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from utils.segmap_utils import label2colormap_torch
from utils.flow_utils import flow_to_image


def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')

    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s]: %(message)s', datefmt='%m/%d %H:%M:%S')
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def format_summary(summary, keys=None, end='.'):
    if keys is None:
        keys = summary.keys()
        summary_data = list(summary.values())
    else:
        summary_data = summary

    metrics_str = ""
    for idx, (data, key) in enumerate(zip(summary_data, keys)):
        if 'time' in key:
            metrics_str += "{}:{:6.4f}s{}".format(key.split('/')[-1], data, 
                ', ' if idx < len(summary_data) - 1 else end)
        elif 'iters' in key or 'length' in key or 'width' in key or 'height' in key:
            metrics_str += "{}:{}{}".format(key.split('/')[-1], data, \
                ', ' if idx < len(summary_data) - 1 else end)
        elif 'memory' in key:
            metrics_str += "{}:{}MB{}".format(key.split('/')[-1], data, \
                ', ' if idx < len(summary_data) - 1 else end)
        else:
            metrics_str += "{}:{:8.6f}{}".format(key.split('/')[-1], data, \
                ', ' if idx < len(summary_data) - 1 else end)

    return metrics_str


class Boarder:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.name = cfgs.run_name
        self.full_path = cfgs.full_path

        self.running_metrics = {}

        self.last_time = None
        self.writer = None
        self.is_main = False

        self.img_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        self.img_sigma = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])

    def start(self):
        self.is_main = True
        if not os.path.isdir(self.full_path):
            os.makedirs(self.full_path)
        logging.info('Enable Tensorboard recording: %s' % self.full_path)

    def stop(self):
        self.is_main = False
        logging.info('Disable Tensorboard recording: %s' % self.full_path)

    def init_writer(self):
        self.writer = SummaryWriter(log_dir=self.full_path)

    def _string_summary(self, prev_steps=None):
        if not self.is_main: 
            return

        if prev_steps is None:
            prev_steps = len(self.running_metrics[list(self.running_metrics.keys())[0]])
            # assert equal number of each metric

        metrics_data = [sum(self.running_metrics[k][-prev_steps:]) / prev_steps for k in self.running_metrics.keys()]
        keys = self.running_metrics.keys()

        metrics_str = format_summary(metrics_data, keys, end=', ')

        latest_time = time.time()
        metrics_str += "time:{:4.1f}s.".format(latest_time - self.last_time)
        self.last_time = latest_time
        return metrics_str

    def _write_summary(self, index, group=None):
        if not self.is_main: 
            return

        if self.writer is None:
            self.init_writer()

        for k in self.running_metrics:
            self.writer.add_scalar(k if group is None else "{}/{}".format(group, k), \
                                   sum(self.running_metrics[k])/len(self.running_metrics[k]), index)

    def _clear_summary(self):
        if not self.is_main: 
            return

        for k in self.running_metrics:
            self.running_metrics[k] = []

    def push(self, metrics, group=None):
        if not self.is_main: 
            return

        if self.last_time is None:
            self.last_time = time.time()

        for key in metrics:
            if group is not None:
                loss_key = "{}/{}".format(group, key)
            else:
                loss_key = key

            if loss_key not in self.running_metrics:
                self.running_metrics[loss_key] = []
            self.running_metrics[loss_key].append(metrics[key])

    def write_summary_board(self, index, group=None):
        if not self.is_main: 
            return

        self._write_summary(index, group)
        self._clear_summary()

    def get_step_string(self, prev_steps=None):
        if not self.is_main: 
            return

        if prev_steps is None:
            prev_steps = self.cfgs.show_summary_every_steps

        return self._string_summary(prev_steps)

    def write_scalar_dict(self, index, dict, group=None):
        if not self.is_main: 
            return

        if self.writer is None:
            self.init_writer()

        for key in dict:
            self.writer.add_scalar(key if group is None else "{}/{}".format(group, key), dict[key], index)

    def write_image(self, index, name, image):
        if not self.is_main: 
            return

        if self.writer is None:
            self.init_writer()
        if 'event' in name:
            image = image[0].detach().cpu()
            mid = image.shape[0] // 2
            image = torch.sum(torch.concat([image[:mid], -1 * image[mid:]]), axis=0)
            image = torch.clip((image - image.min()) / (image.max() - image.min() + 1e-5), 0, 1)
        elif name.endswith('mos'):
            image = image[0][0].detach().cpu()
            image = label2colormap_torch(image)
        elif name.endswith('mask'):
            image = image[0].detach().cpu()
        elif 'flow' in name:
            image = image[0].detach().cpu().numpy()
            image = flow_to_image(image, convert_to_bgr=False)
            image = torch.from_numpy(image).permute(2, 0, 1)
        elif 'image' in name:
            image = image[0].detach().cpu() * self.img_sigma + self.img_mean
        else:
            image = image[0].detach().cpu()

        grid = torchvision.utils.make_grid(image)
        self.writer.add_image(name, grid, index)

    def write_image_dict(self, index, dict, group=None):
        if not self.is_main: 
            return

        if self.writer is None:
            self.init_writer()

        for key in dict:
            if group is not None:
                self.write_image(index, "{}/{}".format(group, key), dict[key])
            else:
                self.write_image(index, key, dict[key])
