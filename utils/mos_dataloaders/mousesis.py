import os
import random
import cv2
import numpy as np
import h5py
import json

import torch
from torch.utils.data import Dataset

import sys
sys.path.append('.')
sys.path.append('utils')

VALIDS = [
          'train',
          'eval',
        #   'test'
           ]




class MouseSIS(Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        self.cfgs = cfgs
        self.root_dir = str(cfgs.root_dir)
        self.split = str(cfgs.split)

        self.seq_length = int(cfgs.seq_length)
        self.min_pixels = int(cfgs.min_pixels)
        assert self.seq_length >= 1

        self.preprocess_dir = os.path.join(self.root_dir, 'emos_preprocess')

        self.bbox_gt_dir = os.path.join(self.root_dir, 'bbox_gt')  # bbox gt file path

        # load bbox gt JSON file
        if self.split != 'test':  
            bbox_gt_file = os.path.join(self.bbox_gt_dir, f'{self.split}.json')
            assert os.path.isfile(bbox_gt_file), f"BBox GT file not found: {bbox_gt_file}"
            with open(bbox_gt_file, 'r') as f:
                self.bbox_gt_data = json.load(f)
        else:
            self.bbox_gt_data = None

        self.has_ev = False
        if hasattr(self.cfgs, 'has_ev') and self.cfgs.has_ev:
            self.has_ev = True
            self.event_bins = cfgs.event_bins
            self.event_polarity = cfgs.event_polarity

        if hasattr(cfgs, 'data_seq'):
            self.seqnames = cfgs.data_seq
            print('for {} seqs only'.format(self.seqnames))
        else:
            self.seqnames = []
            if self.split.startswith('full') or self.split.startswith('train'):
                for seq in VALIDS:
                    seq_folder = os.path.join(self.preprocess_dir, seq)
                    if 'train' in seq and os.path.isdir(seq_folder):
                        subseqs = [os.path.join(seq, sub) for sub in os.listdir(seq_folder)]
                        self.seqnames += subseqs

            if self.split.startswith('full') or self.split.startswith('val') or self.split.startswith('eval') or self.split.startswith('test'):
                for seq in VALIDS:
                    seq_folder = os.path.join(self.preprocess_dir, seq)
                    if 'eval' in seq and os.path.isdir(seq_folder):
                        subseqs = [os.path.join(seq, sub) for sub in os.listdir(seq_folder)]
                        self.seqnames += subseqs
                    if 'test' in seq and os.path.isdir(seq_folder):
                        subseqs = [os.path.join(seq, sub) for sub in os.listdir(seq_folder)]
                        self.seqnames += subseqs

        self.seq_frame_count = {}
        self.indices = []
        for seqname in self.seqnames:
            seq_path = os.path.join(self.preprocess_dir, seqname)
            assert os.path.isdir(seq_path)
            files = sorted([f for f in os.listdir(seq_path) if f.endswith('.hdf5')])
            self.seq_frame_count[int(os.path.basename(seqname).replace("seq", ""))] = len(files) + 1
            total_length = len(files) - self.seq_length
            mov_valid = np.loadtxt(os.path.join(seq_path, 'valid.txt'), dtype=(str))
            mov_valid = dict(zip(mov_valid[:, 0], mov_valid[:, 1].astype(bool)))
            for index in range(total_length):
                valid_flag = True
                for seq_idx in range(self.seq_length):
                    img_name = "img_{}.png".format(int(files[index+seq_idx].split('.')[0]))
                    assert img_name in mov_valid.keys()
                    if not mov_valid[img_name]:
                        valid_flag = False
                        break
                if valid_flag:
                    self.indices.append([seqname, \
                                        [int(files[index+seq_idx].split('.')[0]) \
                                        for seq_idx in range(self.seq_length+1)], \
                                        self.seq_length])

    def __len__(self):
        return len(self.indices)

    def open_preprocess_h5py(self, filename, is_img2=True, is_event=True):
        assert os.path.isfile(filename), '{} not exist!'.format(filename)
        h5file = h5py.File(filename, 'r')
        image1 = np.array(h5file["image1"])
        if self.cfgs.split != 'test':
            mov_seg = np.array(h5file["mov_seg"])
        else:
            mov_seg = None
        image2 = None
        if is_img2:
            image2 = np.array(h5file["image2"])

        event_voxel = None
        if is_event:
            event_voxel = np.array(h5file["event_voxel"])
        
        h5file.close()

        return image1, image2, mov_seg, event_voxel
    
    def compute_iou(self, mask, bboxes):
        """
        Compute IoU between mask and each bbox.
        Args:
            mask (np.ndarray): binary mask, shape (H, W).
            bboxes (torch.Tensor): bounding box tensor, shape (N, 4) in [x_min, y_min, x_max, y_max].
        Returns:
            torch.Tensor: IoU between each bbox and mask, shape (N,).
        """
        mask_area = mask.sum()
        ious = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox.int()
            bbox_mask = np.zeros_like(mask, dtype=np.uint8)
            bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1
            intersection = (mask & bbox_mask).sum()
            union = mask_area + bbox_mask.sum() - intersection
            ious.append(intersection / union if union > 0 else 0.0)
        return torch.tensor(ious, dtype=torch.float32)

    def __getitem__(self, i):

        root = self.root_dir
        seq = self.indices[i][0]
        idxs = self.indices[i][1]
        seq_length = self.indices[i][2]
        data_dict = {'root': root, 'seq': seq, 'index': idxs, 'seq_length': seq_length}
        data_dict['keyname'] = os.path.join(seq, '{0:05d}'.format(idxs[0]))

        images = []
        event_voxels = [] if self.has_ev else None
        # if test set, do not read mov_seg
        if self.cfgs.split != 'test':
            mov_segs = []
        else:
            mov_segs = None

        image1, image2 = None, None

        # store bbox gt for current sample
        gt_bboxes = []

        for i in range(len(idxs)-1):
            idx1 = idxs[i]
            is_last = (i == len(idxs)-2)

            preprocess_path = os.path.join(self.preprocess_dir, seq, '{0:05d}.hdf5'.format(idx1))
            assert os.path.isfile(preprocess_path)
            image1, image2, mov_seg, event_voxel = \
                self.open_preprocess_h5py(preprocess_path, is_img2=is_last, is_event=self.has_ev)

            images.append(image1)
            if is_last:
                images.append(image2)

            if self.cfgs.split != 'test':
                mov_segs.append(mov_seg)

            if self.has_ev:
                event_voxels.append(event_voxel)

            # load bbox gt
            seq_id = os.path.basename(seq)
            frame_id = str(idx1)
            if self.bbox_gt_data is not None:
                if seq_id in self.bbox_gt_data and frame_id in self.bbox_gt_data[seq_id]:
                    frame_bboxes = self.bbox_gt_data[seq_id][frame_id]
                    if frame_bboxes:
                        gt_bboxes.append(torch.tensor(frame_bboxes, dtype=torch.float32))
                    else:
                        gt_bboxes.append(torch.empty((0, 4), dtype=torch.float32))
                else:
                    gt_bboxes.append(torch.empty((0, 4), dtype=torch.float32))

        images = np.concatenate(images, axis=-1)
        data_dict['images'] = images

        if self.has_ev:
            event_voxels = np.dstack(event_voxels)
            data_dict['event_voxels'] = event_voxels

        if self.cfgs.split != 'test':
            # id align
            mov_segs = np.dstack(mov_segs)
            processed_mov_segs = mov_segs.copy()
            act_id = 0
            max_id = mov_segs.max()
            obj_flag = np.zeros(max_id+1)
            aligned_bboxes = [torch.empty((0, 4), dtype=torch.float32) for _ in range(seq_length)]

            for frame in range(seq_length):
                # frame_bboxes = gt_bboxes[frame]

                for obj_id in range(1, max_id+1):
                    if obj_flag[obj_id] != 0:
                        continue
                    count = np.count_nonzero(mov_segs[:, :, frame] == obj_id)
                    if count > 0 and count >= self.min_pixels:
                        act_id += 1
                        processed_mov_segs[mov_segs == obj_id] = act_id
                        obj_flag[obj_id] = 1

                        # Find the bbox with the highest IoU for this mask
                        for frame_idx, frame_bboxes in enumerate(gt_bboxes):
                            if frame_bboxes.numel() > 0:
                                mask = (mov_segs[:, :, frame_idx] == obj_id).astype(np.uint8)
                                if mask.sum() == 0:
                                    continue
                                ious = self.compute_iou(mask, frame_bboxes)
                                max_iou_idx = torch.argmax(ious).item()
                                if ious[max_iou_idx] > 0:  # Ensure IoU is valid
                                    aligned_bbox = frame_bboxes[max_iou_idx]
                                    aligned_bboxes[frame_idx] = torch.cat(
                                        (aligned_bboxes[frame_idx], aligned_bbox.unsqueeze(0))
                                    )
                                else:
                                    # # Raise an exception if IoU is not valid
                                    # raise ValueError(
                                    #     f"Invalid IoU detected in frame {frame_idx} for obj_id {obj_id}. "
                                    #     f"Keyname: {data_dict['keyname']}, "
                                    #     f"IoUs: {ious.tolist()}, "
                                    #     f"Max IoU: {ious[max_iou_idx]}, "
                                    #     f"Mask shape: {mask.shape}, "
                                    #     f"BBoxes: {frame_bboxes}"
                                    # )
                                    # all ious are 0, generate a bbox
                                    y_indices, x_indices = np.where(mask > 0)
                                    if len(y_indices) > 0 and len(x_indices) > 0:
                                        x_min, y_min = x_indices.min(), y_indices.min()
                                        x_max, y_max = x_indices.max(), y_indices.max()
                                        generated_bbox = torch.tensor(
                                            [x_min, y_min, x_max, y_max],
                                            dtype=torch.float32,
                                            device=frame_bboxes.device
                                        )
                                        aligned_bboxes[frame_idx] = torch.cat(
                                            (aligned_bboxes[frame_idx], generated_bbox.unsqueeze(0))
                                        )
                                    else:
                                        raise ValueError(
                                            f"Invalid mask detected in frame {frame_idx} for obj_id {obj_id}. "
                                            f"Keyname: {data_dict['keyname']}, "
                                            f"Mask shape: {mask.shape}, "
                                            f"Mask is empty."
                                        )


                    elif count > 0 and count < self.min_pixels:
                        processed_mov_segs[mov_segs == obj_id] = 0
                        obj_flag[obj_id] = 1
            
            for frame in range(seq_length):
                # check length consistency between processed_mov_segs and aligned_bboxes
                unique_ids = np.unique(processed_mov_segs[:, :, frame])
                unique_ids = unique_ids[unique_ids > 0]  # exclude background ID (0)
                if len(unique_ids) != len(aligned_bboxes[frame]):
                    raise ValueError(
                        f"Mismatch in frame {frame}: "
                        f"processed_mov_segs unique IDs ({len(unique_ids)}) != "
                        f"aligned_bboxes count ({len(aligned_bboxes[frame])}). "
                        f"Frame info: unique IDs = {unique_ids}, "
                        f"aligned_bboxes = {aligned_bboxes[frame]}"
                    )
            
            data_dict['mov_segs'] = processed_mov_segs

            data_dict['gt_bboxes'] = aligned_bboxes

        return data_dict
    
    
