from glob import glob
import numpy as np
import h5py
import os
os.environ["KMP_BLOCKTIME"] = "0"
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import sys
sys.path.append('.')
sys.path.append('utils')
from utils.event_utils import eventsToVoxel


def save_preprocess_h5py(filename, image1, image2, mov_seg, event_voxel=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file = h5py.File(filename, 'w')
    file.create_dataset('image1', data=image1, dtype=np.uint8, compression="lzf")
    file.create_dataset('image2', data=image2, dtype=np.uint8, compression="lzf")
    file.create_dataset('mov_seg', data=mov_seg, dtype=np.uint8, compression="lzf")
    if event_voxel is not None:
        file.create_dataset('event_voxel', data=event_voxel, dtype=np.float32, compression="lzf")
    file.close()


data_path = './datasets/EVIMO'
save_path = './datasets/EVIMO/emos_evimo_preprocess'
min_obj_total_pixels = 400

# seq_folders = sorted([f for f in os.listdir(data_path) if not os.path.isfile(f)])
# seq_folders = ['train/box', 
#                'train/floor', 
#                'train/table', 
#                'train/tabletop',
#                'train/tabletop-egomotion', 
#                'train/wall',
#                'eval/box', 
#                'eval/floor', 
#                'eval/table', 
#                'eval/tabletop',
#                'eval/fast', 
#                'eval/wall',]

seq_folders = ['eval/box', 
               'eval/floor', 
               'eval/table', 
               'eval/tabletop',
               'eval/fast', 
               'eval/wall',]

for sequence in seq_folders:
    seq_data_path = os.path.join(data_path, sequence, 'npz')
    save_sub_path = os.path.join(save_path, sequence)
    npz_filenames = sorted(glob(os.path.join(seq_data_path, "*.npz")))
    for npz_name in npz_filenames:
        npz_base_filename = os.path.splitext(os.path.basename(npz_name))[0]

        save_base_path = os.path.join(save_sub_path, npz_base_filename)
        os.makedirs(save_base_path, exist_ok=True)

        print(sequence, "/", npz_base_filename)
        seq_img_path = os.path.join(data_path, sequence, 'txt', npz_base_filename, 'img')

        npz_file = np.load(npz_name, allow_pickle=True)
        events = npz_file['events'] # N * 4 (t, x, y, p) 
        events_ts = events[:, 0]
        events_p = events[:, 3]
        events_p[events_p==0]=-1
        events = np.stack((events[:, 1], events[:, 2], events_ts, events_p), axis=1)
        frames_meta = npz_file['meta'][np.newaxis][0]['frames']

        masks_data = np.array(npz_file['mask'])

        frame_ts = []
        frame_event_index = []
        frame_images_name = []
        for i in range(len(frames_meta)):
            ts = frames_meta[i]['ts']
            frame_ts.append(ts)
            frame_event_index.append(np.searchsorted(events_ts, ts))
            frame_images_name.append(os.path.join(seq_img_path, frames_meta[i]['classical_frame']))

        start = 0
        end = len(frame_images_name) - 1

        mov_valid = []
        images = []

        for i in range(start, end):
            single_events_np = events[frame_event_index[i]:frame_event_index[i+1]]
            img1_np = cv2.imread(frame_images_name[i], cv2.IMREAD_GRAYSCALE)
            img2_np = cv2.imread(frame_images_name[i+1], cv2.IMREAD_GRAYSCALE)
            mask = masks_data[i]
            height, width = img1_np.shape

            preprocess_file = os.path.join(save_base_path, \
                str(os.path.basename(frame_images_name[i]).split('_')[-1].split('.')[0]).rjust(5, '0') + '.hdf5')
            voxel = eventsToVoxel(single_events_np, num_bins=5, height=height, width=width, \
                                  event_polarity=True)

            total_num = 0
            total_pixels = 0
            unique_inst_ids = np.unique(mask[mask > 0])
            new_mask = np.zeros_like(mask)
            for obj_idx in unique_inst_ids:
                if np.count_nonzero(mask==obj_idx) > 0:
                    total_num += 1
                    total_pixels += np.count_nonzero(mask==obj_idx)

                    new_mask[mask==obj_idx] = total_num

            mov_valid.append(total_num > 0 and total_pixels >= min_obj_total_pixels)
            images.append(os.path.basename(frame_images_name[i]))

            save_preprocess_h5py(preprocess_file, img1_np, img2_np, new_mask, voxel)
            print(preprocess_file)
