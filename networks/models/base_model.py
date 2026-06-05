import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
import torch.distributed as dist

from networks.loss.flow.flow_supervised import flow_supervised_loss
from networks.loss.flow.flow_unsup import flow_unsup_loss
from networks.loss.feature.NTXentLoss import NTXentLossWithWeight
from networks.loss.grl import grl

from utils.distributed_utils import ddp_allgather_with_grads

from PIL import Image
import numpy as np
import cv2


class ConvGN(nn.Module):
    def __init__(self, indim, outdim, kernel_size, gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(indim,
                              outdim,
                              kernel_size,
                              padding=kernel_size // 2)
        self.gn = nn.GroupNorm(gn_groups, outdim)

    def forward(self, x):
        return self.gn(self.conv(x))


class FusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        
        super(FusionModule, self).__init__()
        assert len(in_channels_list) == len(out_channels_list), \
            "The lengths of input channel list and output channel list must be equal"
        
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for in_channels, out_channels in zip(in_channels_list, out_channels_list)
        ])

    def forward(self, features_a, features_b):
        fused_features = [
            fusion_layer(torch.cat([feat_a, feat_b], dim=1))
            for fusion_layer, feat_a, feat_b in zip(self.fusion_layers, features_a, features_b)
        ]
        return fused_features


class FPNHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim=256,
                 shortcut_dims=[24, 32, 96, 1280],
                 flow_branch_split=False,
                 align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.flow_branch_split = flow_branch_split
        if self.flow_branch_split:
            in_dim = in_dim // 2
            shortcut_dims = [dim // 2 for dim in shortcut_dims]
            # hidden_dim = hidden_dim // 2

        self.conv_in = ConvGN(in_dim, hidden_dim, 1)

        self.conv_16x = ConvGN(hidden_dim, hidden_dim, 3)
        self.conv_8x = ConvGN(hidden_dim, hidden_dim // 2, 3)
        self.conv_4x = ConvGN(hidden_dim // 2, hidden_dim // 2, 3)

        self.adapter_16x = nn.Conv2d(shortcut_dims[2], hidden_dim, 1)
        self.adapter_8x = nn.Conv2d(shortcut_dims[1], hidden_dim, 1)
        self.adapter_4x = nn.Conv2d(shortcut_dims[0], hidden_dim // 2, 1)

        self.conv_out = nn.Conv2d(hidden_dim // 2, out_dim, 1)

    def forward(self, x, shortcuts):

        if self.flow_branch_split:
            x = x[:, x.shape[1] // 2:]
            shortcuts[2] = shortcuts[2][:, shortcuts[2].shape[1] // 2:]
            shortcuts[1] = shortcuts[1][:, shortcuts[1].shape[1] // 2:]
            shortcuts[0] = shortcuts[0][:, shortcuts[0].shape[1] // 2:]

        x = F.relu_(self.conv_in(x))
        x = F.relu_(self.conv_16x(self.adapter_16x(shortcuts[2]) + x))

        x = F.interpolate(x,
                          size=shortcuts[1].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = F.relu_(self.conv_8x(self.adapter_8x(shortcuts[1]) + x))

        x = F.interpolate(x,
                          size=shortcuts[0].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = F.relu_(self.conv_4x(self.adapter_4x(shortcuts[0]) + x))

        x = self.conv_out(x)

        return x

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class ModalityTransfer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.translator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, feat_event):
        return self.translator(feat_event)

class BaseModel(nn.Module):
    def __init__(self, cfgs, eval_mode=False):
        super().__init__()
        self.cfgs = cfgs
        self.eval_mode = eval_mode
        self.max_obj_num = cfgs.max_obj_num
        self.decoder_num_queries = cfgs.decoder_num_queries

        self.event_bins = self.cfgs.event_bins * 2 if self.cfgs.event_polarity else self.cfgs.event_bins
        self.projected_feature = cfgs.projected_dim if hasattr(cfgs, 'projected_dim') else 256
        self.flow_branch_split = cfgs.flow_branch_split if hasattr(cfgs, 'flow_branch_split') else False

        self.hiddem_channels = 128 if cfgs.encoder == 'mobilenetv2' else 256
        self.att_heads = [1, 1, 2, 2] if cfgs.encoder == 'mobilenetv2' else [1, 2, 4, 8]

        self.decoder_feat_channels = self.projected_feature
        self.decoder_out_channels = self.projected_feature
        self.max_label_num = 1
        self.pred_score_thr = cfgs.pred_score_thr

        self.loss = None
        self.scalar_summary, self.image_summary = {}, {}

        self.imencoder_tt = None
        self.evencoder_tt = None
        self.imencoder_mt = None
        self.evencoder_mt = None
        self.texture_domain_mapper = None
        self.motion_domain_mapper = None
        self.projector = None
        self.mask_decoder = None
        self.flow_decoder = None
        self.mask_fusion = None


        self.texture_discriminator = None
        self.motion_discriminator = None
        self.da_gamma = None
        self.da_lambda = None
        self.current_step = 0
        self.total_steps = getattr(cfgs, "max_steps", 300000)

        encoder_dims = getattr(cfgs, "encoder_dim", [256, 512, 1024])

        if not eval_mode:
            self.event_to_image_texture = nn.ModuleList([
                ModalityTransfer(c, c) for c in encoder_dims
            ])
            self.event_to_image_motion = nn.ModuleList([
                ModalityTransfer(c, c) for c in encoder_dims
            ])
            self.image_to_event_texture = nn.ModuleList([
                ModalityTransfer(c, c) for c in encoder_dims
            ])
            self.image_to_event_motion = nn.ModuleList([
                ModalityTransfer(c, c) for c in encoder_dims
            ])
        else:
            self.event_to_image_texture = None
            self.event_to_image_motion = None
            self.image_to_event_texture = None
            self.image_to_event_motion = None

        self.texture_fusion = FusionModule(encoder_dims, encoder_dims)
        self.motion_fusion = FusionModule(encoder_dims, encoder_dims)

        # save discriminator outputs
        self.domain_pred_texture_src = None
        self.domain_pred_texture_tgt = None
        self.domain_pred_motion_src = None
        self.domain_pred_motion_tgt = None

        self.loss_names = cfgs.loss
        self.loss_weights = cfgs.loss_weights
        assert len(self.loss_names) == len(self.loss_weights)
        for loss_idx, loss_name in enumerate(self.loss_names):
            if loss_name == 'cls':
                self.loss_cls_dict = dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    reduction='mean',
                    class_weight=[1.0] * (self.max_label_num + 1) + [0.1]
                )
            elif loss_name == 'dice':
                self.loss_dice_dict = dict(
                    type='mmdet.DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    reduction='mean',
                    naive_dice=True,
                    eps=1.0,
                    loss_weight=1.0
                )
            elif loss_name == 'focal':
                self.loss_focal_dict = dict(
                    type='mmdet.FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    reduction='mean',
                    loss_weight=1.0
                )
            elif loss_name == 'bbox':
                self.loss_bbox_dict = dict(
                    type='mmdet.L1Loss',
                    reduction='mean',
                    loss_weight=1.0
                )

            elif loss_name == 'flowl1' or loss_name == 'unflow':
                pass
            elif loss_name == 'contrast':
                if not eval_mode:
                    self.ntxentloss = NTXentLossWithWeight()

                    hch = cfgs.encoder_dim[-1] * 2 # self.hiddem_channels
                    if cfgs.encoder == 'mobilenetv2':
                        self.texture_projector = nn.Sequential(
                            nn.Conv2d(hch // 2, hch // 16, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(hch // 16),
                            nn.ReLU(True),
                            nn.Conv2d(hch // 16, hch // 64, kernel_size=3, stride=2, padding=1),
                        )
                        self.motion_projector = nn.Sequential(
                            nn.Conv2d(hch // 2, hch // 16, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(hch // 16),
                            nn.ReLU(True),
                            nn.Conv2d(hch // 16, hch // 64, kernel_size=3, stride=2, padding=1),
                        )
                    else:
                        self.texture_projector = nn.Sequential(
                            nn.Conv2d(hch // 2, hch // 8, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(hch // 8),
                            nn.ReLU(True),
                            nn.Conv2d(hch // 8, hch // 32, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(hch // 32),
                            nn.ReLU(True),
                            nn.Conv2d(hch // 32, hch // 64, kernel_size=1, stride=1, padding=1),
                        )
                        self.motion_projector = nn.Sequential(
                            nn.Conv2d(hch // 2, hch // 8, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(hch // 8),
                            nn.ReLU(True),
                            nn.Conv2d(hch // 8, hch // 32, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(hch // 32),
                            nn.ReLU(True),
                            nn.Conv2d(hch // 32, hch // 64, kernel_size=1, stride=1, padding=1),
                        )
            elif loss_name == 'adv_loss_texture' or loss_name == 'adv_loss_motion':
                pass
            elif loss_name == 'modality_recon':
                pass
            else:
                raise NotImplementedError("Unsuppored loss func {}".format(loss_name))

        self.init_weights()

    def encoder_project(self, im_feats, ev_feats):

        embs = []
        if len(im_feats) == 3:
            for i in range(len(im_feats)):
                x = self.projector[i](im_feats[i], ev_feats[i])
                embs.append(x)
        elif len(im_feats) == 4: # for mobilenetv2 backbone
            for i in range(2):
                x = self.projector[i](im_feats[i], ev_feats[i])
                embs.append(x)

            B, C, H, W = im_feats[3].shape
            im_feat = F.interpolate(im_feats[3].view(B, C // 4, H * 2, W * 2), \
                                    im_feats[2].shape[-2:], mode='bilinear', align_corners=True)
            ev_feat = F.interpolate(ev_feats[3].view(B, C // 4, H * 2, W * 2), \
                                    ev_feats[2].shape[-2:], mode='bilinear', align_corners=True)
            x = torch.concat([im_feats[2], im_feat, ev_feats[2], ev_feat], dim=1)
            x = self.projector[-1](x)
            embs.append(x)

        return embs

    def get_loss(self):

        self.loss = 0.
        self.scalar_summary = {}
        self.im1_warp = None

        for seq_idx in range(self.pred_length):
            batch_gt_instances = []
            gt_mos_segs_idx = self.inputs['mov_segs'][:, seq_idx].unsqueeze(1)
            gt_bboxes_seq = [self.inputs['gt_bboxes'][batch_idx][seq_idx] for batch_idx in range(self.batch_size)]
            for batch_idx in range(self.batch_size):
                labels, masks, bboxes = [], [], []
                max_instance_num = gt_mos_segs_idx[batch_idx].max()
                index_bias = 1
                for instance_id in range(max_instance_num + 1):
                    label = 0 if instance_id == 0 else 1
                    labels.append(torch.tensor(label, dtype=torch.long, device=gt_mos_segs_idx.device))
                    mask = (gt_mos_segs_idx[batch_idx][0] == instance_id)
                    masks.append(mask)
                    if instance_id == 0 or mask.sum() == 0:
                        bboxes.append(torch.ones((1, 4), dtype=torch.float32, device=gt_mos_segs_idx.device) * -1.0)
                        if instance_id != 0:
                            index_bias += 1
                    else:
                        bboxes.append(gt_bboxes_seq[batch_idx][instance_id - index_bias].unsqueeze(0).to(device=gt_mos_segs_idx.device))
                bboxes = torch.cat(bboxes, dim=0).to(torch.float32)
                
                # check length consistency
                if len(labels) != len(masks) or len(labels) != len(bboxes):
                    raise ValueError(
                        f"Inconsistent lengths detected at seq_idx={seq_idx}, batch_idx={batch_idx}: "
                        f"labels={len(labels)}, masks={len(masks)}, bboxes={len(bboxes)}."
                    )
                assert len(masks) != 0
                labels = torch.stack(labels)
                masks = torch.stack(masks).long()
                batch_gt_instances.append(InstanceData(labels=labels.long(), masks=masks, bboxes=bboxes))
 
            mask_losses = self.mask_decoder.loss_by_feat( \
                self.all_cls_scores_list[seq_idx], self.all_mask_preds_list[seq_idx], \
                    self.all_bbox_preds_list[seq_idx], batch_gt_instances, self.batch_img_metas)

            for loss_idx, loss_name in enumerate(self.loss_names):
                if loss_name == 'cls':
                    loss_cls = mask_losses['loss_cls']
                    self.loss += self.loss_weights[loss_idx] * loss_cls
                    self.scalar_summary.update({
                        'cls': loss_cls.item() \
                            if not 'cls' in self.scalar_summary.keys() \
                                else self.scalar_summary['cls'] + loss_cls.item(),
                    })
                elif loss_name == 'dice':
                    loss_dice = mask_losses['loss_dice'][0]
                    self.loss += self.loss_weights[loss_idx] * loss_dice
                    self.scalar_summary.update({
                        'dice': loss_dice.item() \
                            if not 'dice' in self.scalar_summary.keys() \
                                else self.scalar_summary['dice'] + loss_dice.item(),
                    })
                elif loss_name == 'focal':
                    loss_mask = mask_losses['loss_mask'][0]
                    self.loss += self.loss_weights[loss_idx] * loss_mask
                    self.scalar_summary.update({
                        'focal': loss_mask.item() \
                            if not 'focal' in self.scalar_summary.keys() \
                                else self.scalar_summary['focal'] + loss_mask.item(),
                    })
                elif loss_name == 'bbox':  # bbox loss
                    loss_bbox = mask_losses['loss_bbox'][0]
                    self.loss += self.loss_weights[loss_idx] * loss_bbox
                    self.scalar_summary['bbox'] = self.scalar_summary.get('bbox', 0.) + loss_bbox.item()
                    self.scalar_summary.update({
                        'bbox': loss_bbox.item() \
                            if not 'bbox' in self.scalar_summary.keys() \
                                else self.scalar_summary['bbox'] + loss_bbox.item(),
                    })
                elif loss_name == 'flowl1':
                    loss_flow = flow_supervised_loss(self.pred_flows_list[seq_idx], \
                                                     self.inputs['flow_2d'][:, seq_idx*2:(seq_idx+1)*2])
                    self.loss += self.loss_weights[loss_idx] * loss_flow
                    self.scalar_summary.update({
                        'flowl1': loss_flow.item() \
                            if not 'flowl1' in self.scalar_summary.keys() \
                                else self.scalar_summary['flowl1'] + loss_flow.item(),
                    })
                elif loss_name == 'unflow':
                    if 'raw_images' in self.inputs.keys():
                        image1 = self.inputs['raw_images'][:, seq_idx*3:(seq_idx+1)*3]
                        image2 = self.inputs['raw_images'][:, (seq_idx+1)*3:(seq_idx+2)*3]
                    else:
                        image1 = self.inputs['images'][:, seq_idx*3:(seq_idx+1)*3]
                        image2 = self.inputs['images'][:, (seq_idx+1)*3:(seq_idx+2)*3]
                    smooth_loss, photo_loss, census_loss, im1_warp = \
                        flow_unsup_loss(self.pred_flows_list[seq_idx], image1, image2)
                    if seq_idx == 0:
                        self.im1_warp = im1_warp

                    loss_flow = smooth_loss + photo_loss + census_loss
                    self.loss += self.loss_weights[loss_idx] * loss_flow
                    self.scalar_summary.update({
                        'unflow': loss_flow.item() \
                            if not 'unflow' in self.scalar_summary.keys() \
                                else self.scalar_summary['unflow'] + loss_flow.item(),
                        'unflow_smooth': smooth_loss.item() \
                            if not 'unflow_smooth' in self.scalar_summary.keys() \
                                else self.scalar_summary['unflow_smooth'] + smooth_loss.item(),
                        'unflow_photo': photo_loss.item() \
                            if not 'unflow_photo' in self.scalar_summary.keys() \
                                else self.scalar_summary['unflow_photo'] + photo_loss.item(),
                        'unflow_census': census_loss.item() \
                            if not 'unflow_census' in self.scalar_summary.keys() \
                                else self.scalar_summary['unflow_census'] + census_loss.item(),
                    })
                elif loss_name == 'adv_loss_texture':
                    # texture: img source domain label=1, eve target domain label=0
                    B, _, H, W = self.domain_pred_texture_src.shape
                    device = self.domain_pred_texture_src.device

                    label_src = torch.ones((B, 1, H, W), device=device)
                    label_tgt = torch.zeros((B, 1, H, W), device=device)
                    adv_loss_src = F.binary_cross_entropy_with_logits(self.domain_pred_texture_src, label_src)
                    adv_loss_tgt = F.binary_cross_entropy_with_logits(self.domain_pred_texture_tgt, label_tgt)
                    adv_loss = (adv_loss_src + adv_loss_tgt) * 0.5
                    self.loss += self.loss_weights[loss_idx] * adv_loss
                    self.scalar_summary['adv_texture'] = adv_loss.item()
                
                elif loss_name == 'adv_loss_motion':
                    B, _, H, W = self.domain_pred_motion_src.shape
                    # motion: img source domain label=0, eve target domain label=1
                    device = self.domain_pred_motion_src.device

                    label_src = torch.zeros((B, 1, H, W), device=device)
                    label_tgt = torch.ones((B, 1, H, W), device=device)
                    adv_loss_src = F.binary_cross_entropy_with_logits(self.domain_pred_motion_src, label_src)
                    adv_loss_tgt = F.binary_cross_entropy_with_logits(self.domain_pred_motion_tgt, label_tgt)
                    adv_loss = (adv_loss_src + adv_loss_tgt) * 0.5
                    self.loss += self.loss_weights[loss_idx] * adv_loss
                    self.scalar_summary['adv_motion'] = adv_loss.item()
                elif loss_name == 'contrast':
                    # calculate contrast loss only with multi frames
                    assert self.seq_length > 1
                    if seq_idx == 0:
                        embs_ev = []
                        embs_im = []
                        for cidx in range(self.seq_length):
                            
                            im_texture_feats, im_motion_feats, ev_texture_feats, ev_motion_feats = self.mi_features_list[cidx]
                            im_texture_feats = ddp_allgather_with_grads.apply(im_texture_feats.contiguous())
                            im_motion_feats = ddp_allgather_with_grads.apply(im_motion_feats.contiguous())
                            ev_texture_feats = ddp_allgather_with_grads.apply(ev_texture_feats.contiguous())
                            ev_motion_feats = ddp_allgather_with_grads.apply(ev_motion_feats.contiguous())
                            im_texture_feats = F.normalize(self.texture_projector(im_texture_feats).view(self.batch_size, -1), p=2, dim=-1)
                            im_motion_feats = F.normalize(self.motion_projector(im_motion_feats).view(self.batch_size, -1), p=2, dim=-1)
                            ev_texture_feats = F.normalize(self.texture_projector(ev_texture_feats).view(self.batch_size, -1), p=2, dim=-1)
                            ev_motion_feats = F.normalize(self.motion_projector(ev_motion_feats).view(self.batch_size, -1), p=2, dim=-1)
                            embs_im.append(torch.concat([im_texture_feats, im_motion_feats], dim=0))
                            embs_ev.append(torch.concat([ev_texture_feats, ev_motion_feats], dim=0))
                               
                        embs_im = torch.concat(embs_im, dim=0)
                        embs_ev = torch.concat(embs_ev, dim=0)

                        labels_im = torch.arange(self.batch_size * 2).repeat(self.seq_length)
                        labels_ev = torch.arange(self.batch_size * 2).repeat(self.seq_length)

                        loss_contrast_im = self.ntxentloss(embs_im, labels_im)
                        loss_contrast_ev = self.ntxentloss(embs_ev, labels_ev)
                        loss_contrast = (loss_contrast_im + loss_contrast_ev) * 0.5
                        self.loss += self.loss_weights[loss_idx] * loss_contrast
                        self.scalar_summary.update({
                            'contrast': loss_contrast.item() \
                                if not 'contrast' in self.scalar_summary.keys() \
                                    else self.scalar_summary['contrast'] + loss_contrast.item(),
                        })
                elif loss_name == 'modality_recon':
                    def compute_l2_loss(translated_list, original_list):
                        return sum(
                            F.mse_loss(translated, original.detach())
                            for translated, original in zip(translated_list, original_list)
                        ) / len(translated_list)
                    # compute reconstruction loss for each modality
                    recon_loss_texture_im = compute_l2_loss(self.im_texture_translated, self.im_texture_feats)
                    recon_loss_motion_im = compute_l2_loss(self.im_motion_translated, self.im_motion_feats)
                    recon_loss_texture_ev = compute_l2_loss(self.ev_texture_translated, self.ev_texture_feats)
                    recon_loss_motion_ev = compute_l2_loss(self.ev_motion_translated, self.ev_motion_feats)

                    recon_loss = (
                        recon_loss_texture_im + recon_loss_motion_im +
                        recon_loss_texture_ev + recon_loss_motion_ev
                    ) * 0.25  # average reconstruction loss over four modalities

                    self.loss += self.loss_weights[loss_idx] * recon_loss
                    self.scalar_summary.update({
                        'reconstruction': recon_loss.item() \
                            if 'reconstruction' not in self.scalar_summary else \
                            self.scalar_summary['reconstruction'] + recon_loss.item(),
                    })
                else:
                    raise NotImplementedError("Unsuppored loss func {}".format(loss_name))

        self.scalar_summary.update({
            'loss': self.loss.item(),
        })
        return self.loss

    def encode(self, image1, event_voxel, event_mask=None):
        # 1. four encoders extract features respectively (list)
        self.im_texture_feats = self.imencoder_tt(image1)
        self.im_motion_feats = self.imencoder_mt(image1)
        self.ev_texture_feats = self.evencoder_tt(event_voxel)
        self.ev_motion_feats = self.evencoder_mt(event_voxel)

        contrast_feats = [self.im_texture_feats[-1], self.im_motion_feats[-1], self.ev_texture_feats[-1], self.ev_motion_feats[-1]]

        # 2. concatenate multi-scale features into single feature vectors

        if self.training:
            if self.texture_discriminator is not None:
                im_texture_flat = self.im_texture_feats[-1]
                ev_texture_flat = self.ev_texture_feats[-1]
                self.domain_pred_texture_src = self.texture_discriminator(grl(im_texture_flat, lambd=self.da_lambda))
                self.domain_pred_texture_tgt = self.texture_discriminator(grl(ev_texture_flat, lambd=self.da_lambda))
            if self.motion_discriminator is not None:
                im_motion_flat = self.im_motion_feats[-1]
                ev_motion_flat = self.ev_motion_feats[-1]
                self.domain_pred_motion_src = self.motion_discriminator(grl(im_motion_flat, lambd=self.da_lambda))
                self.domain_pred_motion_tgt = self.motion_discriminator(grl(ev_motion_flat, lambd=self.da_lambda))

            # compute translated features for modality reconstruction loss
            if self.event_to_image_texture is not None:
                self.im_texture_translated = [
                    translator(real) for real, translator in zip(self.ev_texture_feats, self.event_to_image_texture)
                ]
                self.im_motion_translated = [
                    translator(real) for real, translator in zip(self.ev_motion_feats, self.event_to_image_motion)
                ]
                self.ev_texture_translated = [
                    translator(real) for real, translator in zip(self.im_texture_feats, self.image_to_event_texture)
                ]
                self.ev_motion_translated = [
                    translator(real) for real, translator in zip(self.im_motion_feats, self.image_to_event_motion)
                ]

        # fusion mechanism (fuse per scale)
        self.fused_texture_feat = self.texture_fusion(self.im_texture_feats, self.ev_texture_feats)
        self.fused_motion_feat = self.motion_fusion(self.im_motion_feats, self.ev_motion_feats)

        # 5. return fused features
        if (event_mask is not None) and hasattr(self, 'evmask_encoder'):
            ev_mask_feat = self.evmask_encoder(event_mask)
            return self.encoder_project(self.fused_texture_feat, self.fused_motion_feat, ev_mask_feat), contrast_feats
        else:
            return self.encoder_project(self.fused_texture_feat, self.fused_motion_feat), contrast_feats


    def forward(self, inputs, is_Train=False):
        pass

    def output_fusion(self):
        batch_size = self.batch_size

        fused_results = self.mask_fusion.predict( \
            self.all_cls_scores_list[0][-1], self.all_mask_preds_list[0][-1], \
            self.all_bbox_preds_list[0][-1], self.batch_img_metas, rescale=True)

        pred_mos = []
        all_bboxes = []  # store bbox results for each batch
        for batch_idx in range(batch_size):
            instance = fused_results[batch_idx]['ins_results']
            instance = instance[instance.scores > self.pred_score_thr]
            labels = instance.labels
            masks = instance.masks
            # bbox info is stored in instance.bboxes field
            bboxes = instance.bboxes if hasattr(instance, 'bboxes') else None

            mos = torch.zeros(self.image_size, dtype=torch.long, device=self.device)
            instance_idx = 1
            for label_idx in range(len(labels)):
                if labels[label_idx] > 0:
                    mos[masks[label_idx]] = instance_idx
                    instance_idx += 1
            pred_mos.append(mos)
            all_bboxes.append(bboxes)

        self.pred_mos = torch.stack(pred_mos, dim=0).unsqueeze(1)
        self.pred_mask_1d = (self.pred_mos!=0).float()

        output_dict = {
            'image1': self.images[:, :3],
            'event_voxel': self.event_voxels[:, :self.event_bins],
            'pred_mos': self.pred_mos,
            'pred_mask': self.pred_mask_1d,
            'pred_bboxes': all_bboxes  # add bbox info
        }

        if self.images.shape[1] > 3:
            output_dict.update({
                'image2': self.images[:, 3:6],
            })
        if hasattr(self, 'im1_warp') and self.im1_warp is not None:
            output_dict.update({
                'image1_warp': self.im1_warp,
            })
        if hasattr(self, 'pred_flows_list') and len(self.pred_flows_list) > 0:
            output_dict.update({
                'pred_flow': self.pred_flows_list[0],
            })
        if 'flow_2d' in self.inputs.keys():
            output_dict.update({
                'gt_flow': self.inputs['flow_2d'][:, :2],
            })
        if 'mov_segs' in self.inputs.keys():
            gt_mos = self.inputs['mov_segs'][:, 0].unsqueeze(1)
            output_dict.update({
                'gt_mos': gt_mos,
                'gt_mask': (gt_mos!=0).float(),
            })

        return output_dict

    def get_scalar_summary(self):
        return self.scalar_summary

    def get_image_summary(self):
        return self.output_fusion()

    def init_weight(self, module):
        if module is None:
            pass
        elif hasattr(module, 'init_weights'):
            module.init_weights()
        else:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def init_weights(self):

        self.init_weight(self.imencoder_tt)
        self.init_weight(self.evencoder_tt)
        self.init_weight(self.imencoder_mt)
        self.init_weight(self.evencoder_mt)
        self.init_weight(self.mask_decoder)
        self.init_weight(self.projector)
        self.init_weight(self.flow_decoder)

