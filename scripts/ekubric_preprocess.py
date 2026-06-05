import random
import os
import cv2
import h5py
import numpy as np
import torch
import sys
sys.path.append('.')
sys.path.append('utils')
from utils.segmap_utils import load_seg_png
from utils.event_utils import load_events_h5, eventsToVoxel
from utils.flow_utils import load_flow_png


def save_preprocess_h5py(filename, image1, image2, mov_seg, flow_2d, event_voxel=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file = h5py.File(filename, 'w')
    file.create_dataset('image1', data=image1, dtype=np.uint8, compression="lzf")
    file.create_dataset('image2', data=image2, dtype=np.uint8, compression="lzf")
    file.create_dataset('mov_seg', data=mov_seg, dtype=np.uint8, compression="lzf")
    file.create_dataset('flow_2d', data=flow_2d, dtype=np.float32, compression="lzf")
    if event_voxel is not None:
        file.create_dataset('event_voxel', data=event_voxel, dtype=np.float32, compression="lzf")
    file.close()


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

event_bins = 5
event_polarity = True

root = './datasets/ekubric'
preprocess_dir = os.path.join(root, 'emos_ekubric_preprocess')

ls_folder = os.path.join(root, 'rgba')
for seq_idx, seq in enumerate(sorted(os.listdir(ls_folder))):
    seq_path = os.path.join(ls_folder, seq)
    images = sorted([f for f in os.listdir(seq_path)])

    mov_valid = np.loadtxt(os.path.join(root, 'moving_segmentation', seq, 'valid.txt'), dtype=(str))
    mov_valid = dict(zip(mov_valid[:, 0], mov_valid[:, 1].astype(bool)))

    for index in range(len(images)-1):
        idx1 = int(images[index].split('.')[0])
        idx2 = idx1 + 1

        if not mov_valid[images[index]]:
            continue

        img1_path = os.path.join(root, 'rgba', seq, '{0:05d}.png'.format(idx1))
        assert os.path.isfile(img1_path)
        image1 = cv2.imread(img1_path)[..., ::-1]

        img2_path = os.path.join(root, 'rgba', seq, '{0:05d}.png'.format(idx2))
        assert os.path.isfile(img2_path)
        image2 = cv2.imread(img2_path)[..., ::-1]

        seg_path = os.path.join(root, 'moving_segmentation', seq, '{0:05d}.png'.format(idx1))
        assert os.path.isfile(seg_path)
        mov_seg = load_seg_png(seg_path)

        flow_path = os.path.join(root, 'forward_flow', seq, '{0:05d}.png'.format(idx1))
        assert os.path.isfile(flow_path)
        flow_2d, flow_2d_mask = load_flow_png(flow_path)
        assert np.count_nonzero(flow_2d_mask == 0) == 0

        height, width = image1.shape[:2]
        ev_raw_path = os.path.join(root, 'events', seq, '{0:05d}_event.hdf5'.format(idx1))
        assert os.path.isfile(ev_raw_path)
        events = load_events_h5(ev_raw_path)
        event_voxel = eventsToVoxel(events, num_bins=event_bins, height=height, width=width, \
            event_polarity=bool(event_polarity), temporal_bilinear=True)

        preprocess_path = os.path.join(preprocess_dir, seq, '{0:05d}.hdf5'.format(idx1))
        save_preprocess_h5py(preprocess_path, image1, image2, mov_seg, flow_2d, event_voxel)
        print(preprocess_path)
