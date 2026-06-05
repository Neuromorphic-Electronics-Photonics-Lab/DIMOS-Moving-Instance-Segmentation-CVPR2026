import cv2
import png
from PIL import Image
import numpy as np
import colorsys
import torch


def label2colormap(label):
    m = label.astype(np.uint8)
    r, c = m.shape
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3 | (m & 64) >> 1
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2 | (m & 128) >> 2
    cmap[:, :, 2] = (m & 4) << 5 | (m & 32) << 1
    return cmap


def label2colormap_torch(label):
    m = label.int()
    r, c = m.shape
    cmap = torch.zeros((3, r, c), dtype=torch.uint8)
    cmap[0, :, :] = (m & 1) << 7 | (m & 8) << 3 | (m & 64) >> 1
    cmap[1, :, :] = (m & 2) << 6 | (m & 16) << 2 | (m & 128) >> 2
    cmap[2, :, :] = (m & 4) << 5 | (m & 32) << 1
    return cmap


def load_seg_png(filepath, coding='hls'):
    if coding == 'hls':
        img = Image.open(filepath)
        img.load()
        return np.asarray(img.convert('P'))
    else:
        raise NotImplementedError('Unsupported png coding %s' % coding)


def hls_palette(n_colors, first_hue=0.01, lightness=.5, saturation=.7):
    """Get a list of colors where the first is black and the rest are evenly spaced in HSL space."""
    hues = np.linspace(0, 1, int(n_colors) + 1)[:-1]
    hues = (hues + first_hue) % 1
    palette = [(0., 0., 0.)] + [colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues]
    return np.round(np.array(palette) * 255).astype(np.uint8)


def convert_seg_viz(seg_map, object_count=None, coding='hls'):
    if coding == 'hls':
        if object_count == None:
            object_count = seg_map.max()

        palette = hls_palette(object_count + 1)
        return np.asarray(palette[seg_map])
    else:
        raise NotImplementedError('Unsupported png coding %s' % coding)
