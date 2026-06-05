from typing import Any
import torch
import os
import time
import json
import numpy as np

from mmengine.fileio import dump
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils


class CocoEval:
    def __init__(self):
        self.dataset_root = None

        self.metric_items = [
            'coco_mAP', 'coco_mAP_50', 'coco_mAP_75', 'coco_mAP_s', 'coco_mAP_m', 'coco_mAP_l'
        ]
        self.coco_metric_names = {
            'coco_mAP': 0,
            'coco_mAP_50': 1,
            'coco_mAP_75': 2,
            'coco_mAP_s': 3,
            'coco_mAP_m': 4,
            'coco_mAP_l': 5,
            'coco_AR@100': 6,
            'coco_AR@300': 7,
            'coco_AR@1000': 8,
            'coco_AR_s@1000': 9,
            'coco_AR_m@1000': 10,
            'coco_AR_l@1000': 11
        }

        self.imgname2imgid = None
        self.is_init = False
        self.temp_dir = None

        self.pred_results = []

    def init_gt(self):
        assert self.dataset_root is not None
        self.gtfilename = os.path.join(self.dataset_root, 'val_movsegs_cocogt.json')
        coco_gt_json = json.load(open(self.gtfilename))

        self.imgname2imgid = dict()
        for i in range(len(coco_gt_json['images'])):
            img_info = coco_gt_json['images'][i]
            if 'keyname' in img_info.keys():
                keyname = img_info['keyname']
            else:
                filename = img_info['file_name']
                keyname = '/'.join(filename.split('/')[-2:])
            self.imgname2imgid[keyname] = img_info['id']

        self.is_init = True

    def summary(self, temp_dir='./', gpu_idx=0, n_gpu=1):
        self.temp_dir = temp_dir

        dump(self.pred_results, os.path.join(self.temp_dir, 'pred_cocofmt_gpu{}.json'.format(gpu_idx)))

        result_dict = {}
        if gpu_idx == 0:

            full_results = []
            for idx in range(n_gpu):
                path = os.path.join(self.temp_dir, 'pred_cocofmt_gpu{}.json'.format(idx))
                try_times = 5
                is_OK = False
                time.sleep(1)
                for _ in range(try_times):
                    if os.path.isfile(path):
                        full_results += json.load(open(path))
                        is_OK = True
                        break
                    else:
                        time.sleep(1)
                if not is_OK:
                    raise TimeoutError('cannot find {}, stop...'.format(path))
            
            if len(full_results) > 0:
                coco_api = COCO(self.gtfilename)
                coco_dt = coco_api.loadRes(full_results)
                cocoEval = COCOeval(coco_api, coco_dt, "segm")
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()

                for metric_item in self.metric_items:
                    key = f'{metric_item}'
                    val = cocoEval.stats[self.coco_metric_names[metric_item]]
                    result_dict[key] = float(f'{round(val, 3)}')
            else:
                result_dict['coco_mAP'] = 0.

        return result_dict

    def reset(self, gpu_idx=0):
        path = os.path.join(self.temp_dir, 'pred_cocofmt_gpu{}.json'.format(gpu_idx))
        assert os.path.isfile(path)
        os.remove(path)
        self.__init__()

    def __call__(self, outputs, inputs):
        if not self.is_init:
            assert 'root' in inputs.keys()
            self.dataset_root = inputs['root'][0]
            self.init_gt()

        if 'pred_mos2' in outputs.keys():
            index_id = 1
            pred_mos = outputs['pred_mos2']
        else:
            index_id = 0
            pred_mos = outputs['pred_mos']

        if torch.is_tensor(pred_mos):
            pred_mos = pred_mos.detach().cpu().numpy()

        assert len(pred_mos.shape) == 4 # B, seq_len, H, W

        for batch_idx in range(pred_mos.shape[0]):

            mov_seg = pred_mos[batch_idx][0]

            if 'keyname' in inputs.keys():
                keyname = inputs['keyname'][batch_idx]
            else:
                keyname = '/'.join([inputs['seq'][batch_idx], '{0:05d}.png'.format(inputs['index'][index_id][batch_idx])])
            if not keyname in self.imgname2imgid.keys():
                continue

            image_id = self.imgname2imgid[keyname]

            unique_inst_ids = np.unique(mov_seg[mov_seg > 0])
            for inst_id in unique_inst_ids:

                result_json = dict()
                result_json['image_id'] = image_id

                mask = np.asarray(mov_seg == inst_id, dtype=np.uint8, order='F')
                mask_rle = maskUtils.encode(mask[:, :, None])[0]
                # for json encoding
                mask_rle['counts'] = mask_rle['counts'].decode()

                result_json['segmentation'] = mask_rle
                result_json['category_id'] = 1
                result_json['score'] = 1.0

                self.pred_results.append(result_json)

        return dict()
