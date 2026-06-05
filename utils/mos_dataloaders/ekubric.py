import os
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch


class KubricData(Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        self.cfgs = cfgs
        self.root_dir = str(cfgs.root_dir)
        self.split = str(cfgs.split)

        self.seq_length = int(cfgs.seq_length)
        self.min_pixels = int(cfgs.min_pixels)
        assert self.seq_length >= 1

        self.preprocess_dir = os.path.join(self.root_dir, 'emos_preprocess_eb5p1')

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
                seq_file = os.path.join(self.root_dir, 'train_seqs.txt')
                assert os.path.isfile(seq_file)
                self.seqnames = self.seqnames + np.loadtxt(seq_file, dtype=str).tolist()

            if self.split.startswith('full') or self.split.startswith('val'):
                seq_file = os.path.join(self.root_dir, 'val_seqs.txt')
                assert os.path.isfile(seq_file)
                self.seqnames = self.seqnames + np.loadtxt(seq_file, dtype=str).tolist()

        self.indices = []
        unvalid_count = 0
        for seqname in self.seqnames:
            seq_path = os.path.join(self.preprocess_dir, seqname)
            if not os.path.isdir(seq_path):
                continue

            files = sorted([f for f in os.listdir(seq_path)])
            total_length = len(files) - self.seq_length + 1

            mov_valid = np.loadtxt(os.path.join(self.root_dir, 'moving_segmentation', seqname, 'valid.txt'), dtype=(str))
            mov_valid = dict(zip(mov_valid[:, 0], mov_valid[:, 1].astype(bool)))
            for index in range(total_length):
                valid_flag = True
                for seq_idx in range(self.seq_length):
                    current_filename = "{0:05d}.png".format(int(files[index].split('.')[0]) + seq_idx)

                    assert current_filename in mov_valid.keys()
                    if not mov_valid[current_filename] or not os.path.isfile(os.path.join(seq_path, current_filename.split('.')[0] + '.hdf5')):
                        unvalid_count += 1
                        valid_flag = False
                        
                        break
                if valid_flag:
                    self.indices.append([seqname, \
                                         [int(files[index].split('.')[0]) + seq_idx \
                                          for seq_idx in range(self.seq_length+1)], \
                                         self.seq_length])
                    
        print(f"Total unvalid data: {unvalid_count}")

    def __len__(self):
        return len(self.indices)

    def open_preprocess_h5py(self, filename, is_img2=True, is_event=True):
        assert os.path.isfile(filename), '{} not exist!'.format(filename)
        h5file = h5py.File(filename, 'r')
        image1 = np.array(h5file["image1"])
        mov_seg = np.array(h5file["mov_seg"])

        flow_2d = np.array(h5file["flow_2d"])

        image2 = None
        if is_img2:
            image2 = np.array(h5file["image2"])

        event_voxel = None
        if is_event:
            event_voxel = np.array(h5file["event_voxel"])

        return image1, image2, mov_seg, flow_2d, event_voxel

    def save_preprocess_h5py(self, filename, image1, image2, mov_seg, flow_2d, event_voxel=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file = h5py.File(filename, 'w')
        file.create_dataset('image1', data=image1, dtype=np.uint8, compression="lzf")
        file.create_dataset('image2', data=image2, dtype=np.uint8, compression="lzf")
        file.create_dataset('mov_seg', data=mov_seg, dtype=np.uint8, compression="lzf")
        file.create_dataset('flow_2d', data=flow_2d, dtype=np.float32, compression="lzf")
        if event_voxel is not None:
            file.create_dataset('event_voxel', data=event_voxel, dtype=np.float32, compression="lzf")
        file.close()

    def __getitem__(self, i):

        root = self.root_dir
        seq = self.indices[i][0]
        idxs = self.indices[i][1]
        seq_length = self.indices[i][2]
        data_dict = {'root': root, 'seq': seq, 'index': idxs, 'seq_length': seq_length}
        data_dict['keyname'] = '/'.join([seq, '{0:05d}.png'.format(idxs[0])])

        images = []
        event_voxels = [] if self.has_ev else None
        mov_segs = []
        flow_2ds = []
        bboxes = []


        image1, image2 = None, None

        for i in range(len(idxs)-1):
            idx1 = idxs[i]
            idx2 = idxs[i+1]
            is_last = (i == len(idxs)-2)
            mov_seg2 = None

            preprocess_path = os.path.join(self.preprocess_dir, seq, '{0:05d}.hdf5'.format(idx1))
            image1, image2, mov_seg, flow_2d, event_voxel = \
                self.open_preprocess_h5py(preprocess_path, is_img2=is_last, is_event=self.has_ev)

            images.append(image1)
            mov_segs.append(mov_seg)
            flow_2ds.append(flow_2d)
            if is_last:
                images.append(image2)
                if mov_seg2 is not None:
                    mov_segs.append(mov_seg2)

            if self.has_ev:
                event_voxels.append(event_voxel)
            
        # ID align and mask filtering
        mov_segs = np.dstack(mov_segs)
        processed_mov_segs = np.zeros_like(mov_segs, dtype=np.int16)
        act_id = 0
        max_id = mov_segs.max()
        obj_flag = np.zeros(max_id + 1)
        for frame in range(seq_length):
            for obj_id in range(1, max_id + 1):
                if obj_flag[obj_id] != 0:
                    continue
                count = np.count_nonzero(mov_segs[:, :, frame] == obj_id)
                if count > 0 and count >= self.min_pixels:
                    act_id += 1
                    processed_mov_segs[mov_segs == obj_id] = act_id
                    obj_flag[obj_id] = 1
                elif count > 0 and count < self.min_pixels:
                    processed_mov_segs[mov_segs == obj_id] = 0
                    obj_flag[obj_id] = 1

        # Generate bbox from filtered mask
        for frame in range(seq_length):
            unique_ids = np.unique(processed_mov_segs[:, :, frame])
            frame_bboxes = []
            for obj_id in unique_ids:
                if obj_id == 0:  # Skip background
                    continue
                mask = (processed_mov_segs[:, :, frame] == obj_id).astype(np.uint8)
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, y_min = x_indices.min(), y_indices.min()
                    x_max, y_max = x_indices.max(), y_indices.max()
                    frame_bboxes.append([x_min, y_min, x_max, y_max])
            bboxes.append(torch.tensor(frame_bboxes, dtype=torch.float32))

        # Check for inconsistency
        if len(frame_bboxes) != len(unique_ids) - 1:  # Exclude background
            print(f"Inconsistency detected in seq {seq}, frame {frame}: "
                  f"{len(frame_bboxes)} bboxes vs {len(unique_ids) - 1} instances")

        data_dict['images'] = np.concatenate(images, axis=-1)
        data_dict['flow_2d'] = np.concatenate(flow_2ds, axis=-1)

        if self.has_ev:
            event_voxels = np.dstack(event_voxels)
            data_dict['event_voxels'] = event_voxels

        
        data_dict['mov_segs'] = processed_mov_segs
        data_dict['gt_bboxes'] = bboxes

        return data_dict
