import os
import json
from glob import glob
import h5py
from tqdm import tqdm

# dataset root directory
data_root = './datasets/MouseSIS'
# path to save bbox GT
save_root = './datasets/MouseSIS/bbox_gt'

# define splits to process and corresponding annotation files
splits = ['val', 'train']  # if train needs to be processed, add to the list

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
        # construct a dict, key is video id (e.g., "02"), value is all annotations for that video
        ann_dict_global = {}
        for ann in annotations.get('annotations', []):
            vid = ann['video_id']
            if vid not in ann_dict_global:
                ann_dict_global[vid] = []
            ann_dict_global[vid].append(ann)
    else:
        # test set has no annotations
        ann_dict_global = {}

    # initialize dict to store bboxes
    bbox_dict = {}

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
            num_frames = f_seq['images'].shape[0]

        # use ann_dict_global to save annotation info for all instances in this video
        annotations_video = ann_dict_global.get(video_id, [])

        # initialize bbox info for current sequence
        bbox_dict[seq_base] = {}

        # iterate over each frame to extract bbox info
        for i in tqdm(range(num_frames - 1), desc=f"Processing frames for {seq_base}"):
            frame_bboxes = []  # store all instance bboxes for current frame
            for ann in annotations_video:
                # ann['bboxes'] is a list whose length should equal the number of video frames
                assert len(ann['bboxes']) == num_frames
                if len(ann['bboxes']) > i and ann['bboxes'][i] is not None:
                    bbox = ann['bboxes'][i]
                    # check if bbox is valid
                    if len(bbox) == 4:
                        # convert bbox format to [x_min, y_min, x_max, y_max]
                        x_min, y_min, width, height = bbox
                        x_max = x_min + width
                        y_max = y_min + height
                        frame_bboxes.append([x_min, y_min, x_max, y_max])
                    else:
                        # output invalid bbox info
                        print(f"Invalid bbox detected! Video ID: {video_id}, Frame ID: {i}, BBox: {bbox}")
            # save current frame bbox info into dict
            bbox_dict[seq_base][str(i)] = frame_bboxes

    # save bbox info to JSON file
    save_path = os.path.join(save_root, f'{split}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(bbox_dict, f, indent=4)
    print(f"Saved bbox GT for {split} at {save_path}")