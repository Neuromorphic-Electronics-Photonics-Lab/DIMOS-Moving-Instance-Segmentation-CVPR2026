import os
os.environ["KMP_BLOCKTIME"] = "0"
import cv2
cv2.setNumThreads(0)
from PIL import Image
import numpy as np
import random

from omegaconf import DictConfig
import torch
torch.set_num_threads(1)
from torch.utils.data import Dataset
import torchvision.transforms as TF


class AugMosData(Dataset):
    def __init__(self, aug_cfgs: DictConfig, dataset: Dataset):
        self.cfgs = aug_cfgs
        self.enabled = aug_cfgs.enabled
        self.resized = list(aug_cfgs.resized)
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_dict = self.dataset[index]

        # if resized is [xx, yy] and both are positive integers, perform resize
        if isinstance(self.resized, list) and len(self.resized) == 2 and all(isinstance(dim, int) and dim > 0 for dim in self.resized):
            target_size = tuple(self.resized)  # convert to (height, width)
            data_dict = self.resize_data(data_dict, target_size)

        transforms = []
        crop_size = None
        if isinstance(self.cfgs.crop_size, int) and self.cfgs.crop_size != -1:
            crop_size = [self.cfgs.crop_size, self.cfgs.crop_size]
        elif self.cfgs.crop_size != [-1, -1]:
            crop_size = self.cfgs.crop_size

        if hasattr(self.cfgs, 'img_norm'):
            img_norm = self.cfgs.img_norm
        else:
            img_norm = True

        max_obj_num = -1
        if self.cfgs.max_obj_num > 0:
            max_obj_num = self.cfgs.max_obj_num

        if self.enabled:

            if self.cfgs.random_scale.enabled:
                min_scale = self.cfgs.random_scale.min_scale
                max_scale = self.cfgs.random_scale.max_scale
                transforms.append(RandomScale(min_scale, max_scale, crop_size))

            if self.cfgs.random_crop.enabled:
                transforms.append(BalancedRandomCrop(crop_size, max_obj_num=max_obj_num))
            else:
                transforms.append(CenterCrop(crop_size))

            if self.cfgs.random_horizontal_flip.enabled:
                transforms.append(RandomHorizontalFlip(self.cfgs.random_horizontal_flip.prob))

            if self.cfgs.random_vertical_flip.enabled:
                transforms.append(RandomVerticalFlip(self.cfgs.random_vertical_flip.prob))

            if self.cfgs.random_color_jitter.enabled:
                transforms.append(RandomColorJitter(self.cfgs.random_color_jitter.prob))

            if self.cfgs.random_gray.enabled:
                transforms.append(RandomGrayScale(self.cfgs.random_gray.prob))

            if self.cfgs.random_blur.enabled:
                transforms.append(RandomGaussianBlur(self.cfgs.random_blur.prob, \
                                                     self.cfgs.random_blur.kernel_size))

        else:
            if crop_size is not None:
                transforms.append(CenterCrop(crop_size))

        transforms.append(IDAlign(self.cfgs.min_pixels))
        # transforms.append(NormalizeBBox()) 
        transforms.append(ToTensor(is_norm=img_norm))

        transforms = TF.Compose(transforms)
        return transforms(data_dict)
    
    def resize_data(self, data_dict, target_size):
        """
        Resize images, event voxels, masks, and bboxes to the target size.
        Args:
            data_dict (dict): The input data dictionary.
            target_size (tuple): The target size (height, width).
        Returns:
            dict: The resized data dictionary.
        """
        h, w = target_size

        for key in data_dict.keys():
            item = data_dict[key]
            if item is None:
                continue

            if key == 'images':
                # Resize images
                num_frames = item.shape[2] // 3
                self.original_height = item.shape[0]
                self.original_width = item.shape[1]
                resized_frames = []
                for i in range(num_frames):
                    frame = item[:, :, i * 3:(i + 1) * 3]
                    resized_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                    resized_frames.append(resized_frame)
                data_dict[key] = np.concatenate(resized_frames, axis=2)

            elif key == 'event_voxels':
                # Resize event voxels
                num_bins = item.shape[2]
                resized_voxels = []
                for i in range(num_bins):
                    voxel = item[:, :, i]
                    resized_voxel = cv2.resize(voxel, (w, h), interpolation=cv2.INTER_LINEAR)
                    resized_voxels.append(resized_voxel[:, :, None])
                data_dict[key] = np.concatenate(resized_voxels, axis=2)

            elif 'mov_segs' in key:
                # Resize masks
                num_masks = item.shape[2]
                resized_masks = []
                for i in range(num_masks):
                    mask = item[:, :, i]
                    resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    resized_masks.append(resized_mask[:, :, None])
                data_dict[key] = np.concatenate(resized_masks, axis=2)

            elif key == 'gt_bboxes':
                # Resize bboxes
                scale_x = w / self.original_width
                scale_y = h / self.original_height
                resized_bboxes = []
                for frame_idx in range(num_frames-1):
                    bbox = item[frame_idx]
                    if bbox.numel() == 0:  
                        resized_bboxes.append(torch.empty((0, 4), dtype=torch.float32))
                        continue
                    if bbox.dim() != 2 or bbox.size(1) != 4:  # check if bbox dimensions are as expected
                        raise ValueError(f"Invalid bbox shape: {bbox.shape}. Expected [num_instances, 4].")
                    resized_tensor = bbox.clone()  # create a copy to avoid modifying original data
                    resized_tensor[:, 0] *= scale_x  # x_min
                    resized_tensor[:, 1] *= scale_y  # y_min
                    resized_tensor[:, 2] *= scale_x  # x_max
                    resized_tensor[:, 3] *= scale_y  # y_max
                    resized_bboxes.append(resized_tensor)
                data_dict[key] = resized_bboxes

            elif key == 'flow_2d':
                scale_x = w / self.original_width
                scale_y = h / self.original_height
                resized_flows = []
                for i in range(num_frames-1):
                    flow = item[:, :, i * 2:(i + 1) * 2]
                    resized_flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
                    resized_flow[:, :, 0] *= scale_x
                    resized_flow[:, :, 1] *= scale_y
                    resized_flows.append(resized_flow)
                data_dict[key] = np.concatenate(resized_flows, axis=2)
                

        return data_dict


class BalancedRandomCrop(object):
    def __init__(self, crop_size, max_step=10, max_obj_num=10):
        self.crop_size = crop_size
        self.max_step = max_step
        self.max_obj_num = max_obj_num
        self.min_obj_pixel_num = 100

    def __call__(self, data_dict):
        if self.crop_size == None:
            return data_dict

        h, w = data_dict['images'].shape[:2]
        new_h, new_w = self.crop_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w
        if self.crop_size == [h, w]:
            return data_dict

        step = 0
        is_contain_obj = False
        label_key = 'mov_segs' if 'mov_segs' in data_dict.keys() else 'vos_segs'

        # select valid start point
        while (not is_contain_obj) and (step < self.max_step):
            step += 1
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            after_crop = []
            contains = []

            for idx in range(data_dict[label_key].shape[2]):
                elem = data_dict[label_key][:, :, idx]
                tmp = elem[top:top + new_h, left:left + new_w]
                contains.append(np.unique(tmp))
                after_crop.append(tmp)

            all_obj = list(np.sort(contains[0]))

            if all_obj[-1] == 0:
                continue

            # remove background
            if all_obj[0] == 0:
                all_obj = all_obj[1:]

            # remove small obj
            new_all_obj = []
            for obj_id in all_obj:
                after_crop_pixels = np.sum(after_crop[0] == obj_id)
                if after_crop_pixels > self.min_obj_pixel_num:
                    new_all_obj.append(obj_id)

            if len(new_all_obj) == 0:
                is_contain_obj = False
            else:
                is_contain_obj = True

            if self.max_obj_num > 0 and len(new_all_obj) > self.max_obj_num:
                random.shuffle(new_all_obj)
                new_all_obj = new_all_obj[:self.max_obj_num]

            all_obj = [0] + new_all_obj

        # crop segs
        post_process = []
        for elem in after_crop:
            new_elem = elem * 0
            for idx in range(len(all_obj)):
                obj_id = all_obj[idx]
                if obj_id == 0:
                    continue
                mask = elem == obj_id

                new_elem += (mask * idx).astype(np.uint8)
            post_process.append(new_elem.astype(np.uint8))

        data_dict[label_key] = np.dstack(post_process)

        if 'gt_bboxes' in data_dict:
            # fix bbox gt cropping logic
            cropped_bboxes = []
            for frame_idx, bbox in enumerate(data_dict['gt_bboxes']):
                if bbox.numel() == 0:
                    cropped_bboxes.append(torch.empty((0, 4), dtype=torch.float32))
                    continue
                cropped_bbox = bbox.clone()
                cropped_bbox[:, 0] = torch.clamp(bbox[:, 0] - left, min=0, max=new_w)
                cropped_bbox[:, 1] = torch.clamp(bbox[:, 1] - top, min=0, max=new_h)
                cropped_bbox[:, 2] = torch.clamp(bbox[:, 2] - left, min=0, max=new_w)
                cropped_bbox[:, 3] = torch.clamp(bbox[:, 3] - top, min=0, max=new_h)
                valid_mask = (cropped_bbox[:, 2] > cropped_bbox[:, 0]) & (cropped_bbox[:, 3] > cropped_bbox[:, 1])
                cropped_bboxes.append(cropped_bbox[valid_mask])
            data_dict['gt_bboxes'] = cropped_bboxes

        # crop rest
        for key in data_dict.keys():
            item = data_dict[key]
            if item is None:
                continue
            if not isinstance(item, np.ndarray):
                continue
            if 'segs' in key:
                continue

            if key == 'images' or key == 'event_voxels' or key == 'flow_2d':
                data_dict[key] = item[top:top + new_h, left:left + new_w]
            else:
                raise NotImplementedError('Not Implemented data in aug, key: {}'.format(key))
        


        return data_dict


class CenterCrop(object):
    def __init__(self, crop_size, max_obj_num=10):
        self.crop_size = crop_size
        self.max_obj_num = max_obj_num # TODO not Implemented

    def __call__(self, data_dict):
        if self.crop_size == None:
            return data_dict

        h, w = data_dict['images'].shape[:2]
        if self.crop_size == [h, w]:
            return data_dict

        new_h, new_w = self.crop_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        for key in data_dict.keys():
            item = data_dict[key]
            if item is None:
                continue
            if not isinstance(item, np.ndarray):
                continue

            if key == 'images' or key == 'event_voxels' or 'segs' in key or key == 'flow_2d':
                data_dict[key] = item[top:top + new_h, left:left + new_w]
            else:
                raise NotImplementedError('Not Implemented data in aug, key: {}'.format(key))
        
        if 'gt_bboxes' in data_dict:
            cropped_bboxes = []
            for frame_idx, bbox in enumerate(data_dict['gt_bboxes']):
                if bbox.numel() == 0:
                    cropped_bboxes.append(torch.empty((0, 4), dtype=torch.float32))
                    continue
                cropped_bbox = bbox.clone()
                cropped_bbox[:, 0] = torch.clamp(bbox[:, 0] - left, min=0, max=new_w)
                cropped_bbox[:, 1] = torch.clamp(bbox[:, 1] - top, min=0, max=new_h)
                cropped_bbox[:, 2] = torch.clamp(bbox[:, 2] - left, min=0, max=new_w)
                cropped_bbox[:, 3] = torch.clamp(bbox[:, 3] - top, min=0, max=new_h)
                valid_mask = (cropped_bbox[:, 2] > cropped_bbox[:, 0]) & (cropped_bbox[:, 3] > cropped_bbox[:, 1])
                cropped_bboxes.append(cropped_bbox[valid_mask])
            data_dict['gt_bboxes'] = cropped_bboxes

        return data_dict


class RandomScale(object):
    def __init__(self, min_scale=0.7, max_scale=1.3, crop_size=None):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.crop_size = crop_size

    def __call__(self, data_dict):
        if self.crop_size is not None:
            h, w = data_dict['images'].shape[:2]
            min_scale_h = max(self.min_scale, self.crop_size[0] / h)
            min_scale_w = max(self.min_scale, self.crop_size[1] / w)
        else:
            min_scale_h = self.min_scale
            min_scale_w = self.min_scale

        sc_x = np.random.uniform(min_scale_w, self.max_scale)
        sc_y = np.random.uniform(min_scale_h, self.max_scale)

        for key in data_dict.keys():
            item = data_dict[key]
            if item is None:
                continue
            if not isinstance(item, np.ndarray):
                continue

            if key == 'images' or key == 'event_voxels':
                data_dict[key] = cv2.resize(
                    item, dsize=None, fx=sc_x, fy=sc_y,
                    interpolation=cv2.INTER_NEAREST,
                )
            elif 'flow' in key:
                flow_2d = cv2.resize(
                    item, dsize=None, fx=sc_x, fy=sc_y,
                    interpolation=cv2.INTER_NEAREST,
                )
                data_dict[key] = flow_2d * ([sc_x, sc_y] * data_dict['seq_length'])
            elif 'segs' in key:
                item = cv2.resize(
                    item, dsize=None, fx=sc_x, fy=sc_y,
                    interpolation=cv2.INTER_NEAREST,
                )
                if len(item.shape) == 2:
                    data_dict[key] = item[:, :, None]
                else:
                    data_dict[key] = item

        if 'gt_bboxes' in data_dict:
            scaled_bboxes = []
            for frame_idx, frame_bboxes in enumerate(data_dict['gt_bboxes']):
                # frame_bboxes: Tensor[num_inst,4]
                if frame_bboxes.numel() == 0:
                    scaled_bboxes.append(torch.empty((0, 4), dtype=torch.float32))
                    continue
                bbox = frame_bboxes.clone()
                # x_min/x_max multiplied by sc_x, y_min/y_max multiplied by sc_y
                bbox[:, [0, 2]] *= sc_x
                bbox[:, [1, 3]] *= sc_y
                scaled_bboxes.append(bbox)
            data_dict['gt_bboxes'] = scaled_bboxes

        return data_dict


class RandomImageContentAug(object):
    def __init__(self, prob=0.5, unified=False):
        self.prob = prob
        self.unified_multiframe = unified
        self.aug = None

    def __call__(self, data_dict):
        if self.aug is None:
            return data_dict

        if not 'raw_images' in data_dict.keys():
            data_dict['raw_images'] = data_dict['images'].copy()

        img_length = data_dict['images'].shape[2] // 3
        if self.unified_multiframe:
            flag = random.random() < self.prob
            aug_flag = [flag for _ in range(img_length)]
        else:
            aug_flag = [random.random() < self.prob for _ in range(img_length)]

        tmp_list = []
        for idx in range(img_length):
            tmp = data_dict['images'][:, :, (idx*3):(idx+1)*3]
            if aug_flag[idx]:
                tmp = self.aug(tmp)
            tmp_list.append(tmp)
        data_dict['images'] = np.concatenate(tmp_list, axis=2)

        return data_dict


class RandomColorJitter(RandomImageContentAug):
    def __init__(self, prob=0.2):
        super().__init__(prob=prob, unified=False)

        brightness = 0.4
        contrast = 0.4
        saturation = 0.4
        hue = 0.4
        self.aug = self.to_Image_aug
        self.func = TF.ColorJitter(brightness, contrast, saturation, hue)

    def to_Image_aug(self, x):
        x = Image.fromarray(np.uint8(x))
        x = self.func(x)
        return np.array(x, dtype=np.float32)


class RandomGrayScale(RandomColorJitter):
    def __init__(self, prob=0.2):
        super().__init__(prob)
        self.func = TF.Grayscale(num_output_channels=3)


class RandomGaussianBlur(RandomColorJitter):
    def __init__(self, prob=0.2, kernel_size=5):
        super().__init__(prob)
        self.func = TF.GaussianBlur(kernel_size)


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.2, flip_code=1):
        # flip_code = 0 for Vertical, 1 for Horizontal
        self.prob = prob
        self.flip_code = flip_code

    def __call__(self, data_dict):
        flip_flag = random.random() < self.prob
        if not flip_flag:
            return data_dict
        
        h, w = data_dict['images'].shape[:2]

        for key in data_dict.keys():
            item = data_dict[key]
            if item is None:
                continue
            if not isinstance(item, np.ndarray):
                continue

            tmp_list = []
            if key == 'images':
                num = item.shape[2] // 3
                for idx in range(num):
                    tmp = item[:, :, (idx*3):(idx+1)*3]
                    tmp = cv2.flip(tmp, flipCode=self.flip_code)
                    tmp_list.append(tmp)
                data_dict[key] = np.concatenate(tmp_list, axis=2)
            elif 'segs' in key or key == 'event_voxels':
                num = item.shape[2] // 1
                for idx in range(num):
                    tmp = item[:, :, idx:idx+1]
                    tmp = cv2.flip(tmp, flipCode=self.flip_code)
                    tmp_list.append(tmp[:, :, None])
                data_dict[key] = np.concatenate(tmp_list, axis=2)
            elif 'flow' in key:
                for idx in range(0, item.shape[-1], 2):
                    tmp = item[:, :, idx:idx+2]
                    tmp = cv2.flip(tmp, flipCode=self.flip_code)
                    tmp *= [-1.0, 1.0] if self.flip_code == 1 else [1.0, -1.0]
                    tmp_list.append(tmp)
                data_dict[key] = np.concatenate(tmp_list, axis=2)
            else:
                raise NotImplementedError('Not Implemented data in aug, key: {}'.format(key))
        
        if 'gt_bboxes' in data_dict:
                # Flip bboxes
                flipped_bboxes = []
                for frame_idx, bboxes in enumerate(data_dict['gt_bboxes']):
                    if bboxes.numel() == 0:
                        flipped_bboxes.append(torch.empty((0, 4), dtype=torch.float32))
                        continue
                    flipped_bbox = bboxes.clone()
                    if self.flip_code == 1:
                        flipped_bbox[:, 0] = w - bboxes[:, 2]  # Flip x_min
                        flipped_bbox[:, 2] = w - bboxes[:, 0]  # Flip x_max
                    elif self.flip_code == 0:
                        flipped_bbox[:, 1] = h - bboxes[:, 3]
                        flipped_bbox[:, 3] = h - bboxes[:, 1]  # Flip y_min
                    flipped_bboxes.append(flipped_bbox)
                data_dict['gt_bboxes'] = flipped_bboxes

        return data_dict


class RandomVerticalFlip(RandomHorizontalFlip):
    def __init__(self, prob=0.2):
        super().__init__(prob, flip_code=0)


class IDAlign(object):
    def __init__(self, min_pixels=100, max_obj_num=-1):
        self.min_pixels = min_pixels
        self.max_obj_num = max_obj_num

    def __call__(self, data_dict):
        for key in data_dict.keys():
            item = data_dict[key]
            if item is None:
                continue
            if not isinstance(item, np.ndarray):
                continue

            if 'images' in key or key == 'event_voxels' or key == 'flow_2d':
                pass
            elif 'segs' in key:
                # id align and resort
                processed_mov_segs = item.copy()
                act_id = 0
                max_id = item.max()
                act_length = item.shape[-1]
                obj_flag = np.zeros(max_id+1)
                aligned_bboxes = [torch.empty((0, 4), dtype=torch.float32) for _ in range(act_length)]

                gt_bboxes = data_dict.get('gt_bboxes', []) if 'gt_bboxes' in data_dict else None
                for frame in range(act_length):
                    obj_pixels = [np.count_nonzero(item[:, :, frame] == i) for i in range(1, max_id+1)]
                    sorted_obj_idxs = np.flipud(np.argsort(obj_pixels))
                    if self.max_obj_num > 0:
                        sorted_obj_idxs = sorted_obj_idxs[:self.max_obj_num]

                    for obj_idx in sorted_obj_idxs:
                        if obj_flag[obj_idx+1] != 0:
                            continue
                        count = obj_pixels[obj_idx]
                        if count > 0 and count >= self.min_pixels:
                            act_id += 1
                            processed_mov_segs[item == obj_idx+1] = act_id
                            obj_flag[obj_idx+1] = 1
                            
                            # Find the bbox with the highest IoU for this mask
                            for frame_idx, frame_bboxes in enumerate(gt_bboxes):
                                if frame_bboxes.numel() > 0:
                                    mask = (item[:, :, frame_idx] == obj_idx + 1).astype(np.uint8)
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
                                        #     f"Invalid IoU detected in frame {frame_idx} for obj_id {obj_idx}. "
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
                                                f"Invalid mask detected in frame {frame_idx} for obj_id {obj_idx}. "
                                                f"Keyname: {data_dict['keyname']}, "
                                                f"Mask shape: {mask.shape}, "
                                                f"Mask is empty."
                                            )

                        elif count > 0 and count < self.min_pixels:
                            processed_mov_segs[item == obj_idx+1] = 0
                            obj_flag[obj_idx+1] = 1
                    
                for frame in range(act_length):
                    # check length consistency between processed_mov_segs and aligned_bboxes
                    unique_ids = np.unique(processed_mov_segs[:, :, frame])
                    unique_ids = unique_ids[unique_ids > 0]  # exclude background ID (0)
                    if len(unique_ids) != len(aligned_bboxes[frame]):
                        print(f"Mismatch detected in frame {frame}: "
                            f"processed_mov_segs unique IDs ({len(unique_ids)}) != "
                            f"aligned_bboxes count ({len(aligned_bboxes[frame])}). "
                            f"Attempting to fix...")
                        
                        # backup current frame aligned_bboxes
                        backup_bboxes = aligned_bboxes[frame].clone() if len(aligned_bboxes[frame]) > 0 else None
                        # reinitialize current frame aligned_bboxes
                        aligned_bboxes[frame] = torch.empty((0, 4), dtype=torch.float32)
                        # iterate over each unique_id to generate corresponding bbox
                        for obj_id in unique_ids:
                            mask = (processed_mov_segs[:, :, frame] == obj_id).astype(np.uint8)
                            if mask.sum() > 0:  # ensure mask is non-empty
                                if backup_bboxes is not None and len(backup_bboxes) > 0:
                                    # compute IoU and find the most similar bbox
                                    ious = self.compute_iou(mask, backup_bboxes)
                                    max_iou_idx = torch.argmax(ious).item()
                                    if ious[max_iou_idx] > 0:  # if IoU is valid, assign the most similar bbox
                                        aligned_bbox = backup_bboxes[max_iou_idx]
                                        aligned_bboxes[frame] = torch.cat(
                                            (aligned_bboxes[frame], aligned_bbox.unsqueeze(0))
                                        )
                                    else:
                                        # if max IoU is 0, use mask2bbox to assign bbox
                                        y_indices, x_indices = np.where(mask > 0)
                                        x_min, y_min = x_indices.min(), y_indices.min()
                                        x_max, y_max = x_indices.max(), y_indices.max()
                                        generated_bbox = torch.tensor(
                                            [x_min, y_min, x_max, y_max],
                                            dtype=torch.float32,
                                            device=aligned_bboxes[frame].device
                                        )
                                        aligned_bboxes[frame] = torch.cat(
                                            (aligned_bboxes[frame], generated_bbox.unsqueeze(0))
                                        )
                                else:
                                    # if no existing bbox, directly use mask2bbox to assign bbox
                                    y_indices, x_indices = np.where(mask > 0)
                                    x_min, y_min = x_indices.min(), y_indices.min()
                                    x_max, y_max = x_indices.max(), y_indices.max()
                                    generated_bbox = torch.tensor(
                                        [x_min, y_min, x_max, y_max],
                                        dtype=torch.float32,
                                        device=aligned_bboxes[frame].device
                                    )
                                    aligned_bboxes[frame] = torch.cat(
                                        (aligned_bboxes[frame], generated_bbox.unsqueeze(0))
                                    )
                            else:
                                raise ValueError(
                                    f"Invalid mask detected for obj_id {obj_id} in frame {frame}. "
                                    f"Mask is empty."
                                )
                    
                data_dict[key] = processed_mov_segs
                if 'gt_bboxes' in data_dict:
                    data_dict['gt_bboxes'] = aligned_bboxes
            else:
                raise NotImplementedError('Not Implemented data in aug, key: {}'.format(key))

        return data_dict
    
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
    
    


class ToTensor(object):
    def __init__(self, is_norm=True):
        self.is_norm = is_norm

    def __call__(self, data_dict):
        data_dict['img_norm'] = self.is_norm

        for key in data_dict.keys():
            item = data_dict[key]
            if item is None:
                continue
            if not isinstance(item, np.ndarray):
                continue

            if 'images' in key:
                if self.is_norm:
                    num = item.shape[2] // 3
                    item = item.astype(np.float32) / 255.
                    item -= np.tile([0.485, 0.456, 0.406], num)
                    item /= np.tile([0.229, 0.224, 0.225], num)
                item = item.transpose((2, 0, 1))
            elif key == 'event_voxels':
                # item = item / (item.max() / 2. + 1e-5)
                item = item / (item.max() + 1e-5)
                item = item.transpose((2, 0, 1))
            elif 'segs' in key or key == 'flow_2d':
                item = item.transpose((2, 0, 1))
            else:
                raise NotImplementedError('Not Implemented data in aug, key: {}'.format(key))

            data_dict[key] = torch.from_numpy(item)

        return data_dict
    

class NormalizeBBox(object):
    def __call__(self, data_dict):
        """
        Normalize bbox gt.
        Args:
            data_dict (dict): data dictionary containing 'gt_bboxes' and 'images'.
        Returns:
            dict: normalized data dictionary.
        """
        if 'gt_bboxes' in data_dict and 'images' in data_dict:
            h, w = data_dict['images'].shape[:2]
            normalized_bboxes = []
            for bbox in data_dict['gt_bboxes']:
                normalized_bbox = bbox.clone()
                normalized_bbox[:, 0] /= w  # normalize x_min
                normalized_bbox[:, 1] /= h  # normalize y_min
                normalized_bbox[:, 2] /= w  # normalize x_max
                normalized_bbox[:, 3] /= h  # normalize y_max
                normalized_bboxes.append(normalized_bbox)
            data_dict['gt_bboxes'] = normalized_bboxes
        return data_dict

