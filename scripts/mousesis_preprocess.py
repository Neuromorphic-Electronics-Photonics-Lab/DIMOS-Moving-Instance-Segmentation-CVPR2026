import os
import json
from glob import glob
import numpy as np
import h5py
import cv2
from tqdm import tqdm

# disable OpenCV multi-threading and OpenCL
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import sys
sys.path.append('.')
sys.path.append('utils')
# import event conversion tools (same as in evimo)
from utils.event_utils import eventsToVoxel

def rle_to_mask(rle, height, width):
    '''Convert a run-length encoded representation of the mask to a binary mask.'''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T

def save_preprocess_h5py(filename, image1, image2, mov_seg, event_voxel=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, 'w') as file:
        file.create_dataset('image1', data=image1, dtype=np.uint8, compression="lzf")
        file.create_dataset('image2', data=image2, dtype=np.uint8, compression="lzf")
        # only save label data when mov_seg is non-empty
        if mov_seg is not None:
            file.create_dataset('mov_seg', data=mov_seg, dtype=np.uint8, compression="lzf")
        if event_voxel is not None:
            file.create_dataset('event_voxel', data=event_voxel, dtype=np.float32, compression="lzf")

# MouseSIS dataset root directory (assumed to be ./MouseSIS/)
data_root = './datasets/MouseSIS'
# path to save preprocessed data
save_root = './datasets/MouseSIS/emos_preprocess'
min_obj_total_pixels = 400

# define splits to process and corresponding annotation files (test annotations may not be available)
splits = ['train', 'val']  # if test is needed, process separately (empty mask can be generated if no mask annotations)

valid_txt_false_lines = []

# iterate over each split
for split in splits:
    split_path = os.path.join(data_root, 'top', split)
    # sequence files under each split (h5 format)
    seq_files = sorted(glob(os.path.join(split_path, 'seq*.h5')))
    # if current split is not test, load annotations
    if split != 'test':
        ann_file = os.path.join(data_root, f'{split}_annotations.json')
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        # construct a dict, key is video id (e.g., "02"), value is all annotations for that video (MS COCO format)
        ann_dict_global = {}
        for ann in annotations.get('annotations', []):
            vid = ann['video_id']
            if vid not in ann_dict_global:
                ann_dict_global[vid] = []
            ann_dict_global[vid].append(ann)
    else:
        # test set has no annotations
        ann_dict_global = {}
    
    # for each sequence file
    for seq_file in seq_files:
        seq_base = os.path.splitext(os.path.basename(seq_file))[0]  # e.g., seq_02
        # extract video id (assuming fixed format, e.g., "seq_02" -> "02")
        if '_' in seq_base:
            video_id = seq_base.split('_')[-1]
        else:
            video_id = seq_base.replace('seq', '')
        print(f'Processing {split} video {seq_base}')
        # open sequence hdf5 file
        with h5py.File(seq_file, 'r') as f_seq:
            images = f_seq['images'][:]       # shape: [num_frames, 720, 1280, 3]
            img2event = f_seq['img2event'][:]   # length is num_frames
            t_all = f_seq['t'][:]              # timestamps of all events
            x_all = f_seq['x'][:]
            y_all = f_seq['y'][:]
            p_all = f_seq['p'][:]
            # get image timestamps (optional, if key 'img_ts' exists)
            if 'img_ts' in f_seq:
                img_ts = f_seq['img_ts'][:]
            else:
                img_ts = None

        num_frames = images.shape[0]
        # note: each frame has different masks, obtained from the segmentation list of each instance in annotations
        if split != 'test':
            # use ann_dict_global to save annotation info for all instances in this video
            annotations_video = ann_dict_global.get(video_id, [])
        else:
            annotations_video = []  # test set has no annotations

        # initialize valid.txt content
        valid_txt_lines = []
        
        # process consecutive frame pairs sequentially: each sample includes images of frame i and frame i+1, corresponding events, and mask (same mask for all frames)
        for i in tqdm(range(num_frames - 1), desc="Processing frames"):
            # get two frames
            image1 = images[i]
            image2 = images[i + 1]
            # defined by img2event: events from img2event[i] to img2event[i+1]
            start_idx = int(img2event[i])
            end_idx = int(img2event[i + 1]) if i + 1 < len(img2event) else len(t_all)
            single_events = np.stack((x_all[start_idx:end_idx],
                                      y_all[start_idx:end_idx],
                                      t_all[start_idx:end_idx],
                                      p_all[start_idx:end_idx]), axis=1)  # shape: [num_events, 4], format: x,y,t,p

            # convert single event segment to voxel representation
            voxel = eventsToVoxel(single_events, num_bins=5, height=720, width=1280, event_polarity=True)

            if split != 'test':
                # get mask from each annotation's segmentation according to current frame i
                mask_frame = np.zeros((720, 1280), dtype=np.uint8)
                total_obj = 0
                total_pixels = 0
                for ann in annotations_video:
                    # ann['segmentations'] is a list whose length should equal the number of video frames
                    assert len(ann['segmentations']) == num_frames
                    if len(ann['segmentations']) > i and ann['segmentations'][i] is not None:
                        seg = ann['segmentations'][i]
                        rle = seg['counts']
                        size = seg['size']  # [height, width]
                        mask_inst = rle_to_mask(rle, size[0], size[1])
                        if np.count_nonzero(mask_inst) > 0:
                            total_obj += 1
                            total_pixels += np.count_nonzero(mask_inst)
                            # assign a new id to this instance (ensure no conflict between different instances)
                            mask_frame[mask_inst == 1] = total_obj
                valid = (total_obj > 0) and (total_pixels >= min_obj_total_pixels)
            else:
                # test set: no mask label info, directly set to empty, and consider valid as True
                mask_frame = None
                valid = True
            # construct save path: save as save_root/<split>/<seq_base>/<frame_index>.hdf5
            if split == 'val': 
                new_split = 'eval'
            else: 
                new_split = split
            save_dir = os.path.join(save_root, new_split, seq_base)
            os.makedirs(save_dir, exist_ok=True)
            sample_filename = os.path.join(save_dir, str(i).zfill(5) + '.hdf5')
            # save sample (note: image1, image2 are raw images; new_mask and voxel are also saved)
            save_preprocess_h5py(sample_filename, image1, image2, mask_frame, voxel)
            print("Saved:", sample_filename, "Valid:", valid)

            # add valid info to valid.txt content
            valid_txt_lines.append(f"img_{i}.png {int(valid)}")
            if not valid:
                valid_txt_false_lines.append(f"img_{i}.png {int(valid)}")
        # save valid.txt file
        valid_txt_path = os.path.join(save_dir, "valid.txt")
        with open(valid_txt_path, 'w') as valid_file:
            valid_file.write("\n".join(valid_txt_lines))
        print(f"Saved valid.txt for {seq_base} at {valid_txt_path}")
print(len(valid_txt_false_lines), "invalid samples found")