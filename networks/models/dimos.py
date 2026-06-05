import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from networks.models.base_model import BaseModel, FPNHead
from networks.encoders.mmdet_encoders import build_mmdet_encoder
from networks.fusion.restormer_arch import CrossTransformerBlock2D
from networks.decoders.maskformer_head import MaskFormerHead
from networks.decoders.maskformer_fusion_head import MaskFormerFusionHead

class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else hidden_dim,
                hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, H, W]
        return self.net(x)  # output [B, 1, H, W]


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, feat_channels, num_heads=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(True),
            nn.Conv2d(feat_channels, out_channels//2, kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(True),
            nn.Conv2d(feat_channels, out_channels//2, kernel_size=1),
        )

        self.att_texture = CrossTransformerBlock2D(out_channels//2, num_heads=num_heads)
        self.att_motion = CrossTransformerBlock2D(out_channels//2, num_heads=num_heads)

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x1, x2, ev_mask_feat):
        x1, x2 = self.conv1(x1), self.conv2(x2)

        mask = F.interpolate(ev_mask_feat, size=x1.shape[-2:], \
                             mode="bilinear", align_corners=True)

        x3 = x2 * mask
        xt = self.att_texture(x1, x3) + x1
        xm = self.att_motion(x3, x1) + x3

        return self.conv3(torch.concat([xt, xm], dim=1))


class DIMOS(BaseModel):
    def __init__(self, cfgs, eval_mode=False):
        super().__init__(cfgs, eval_mode=eval_mode)

        self.imencoder_tt = build_mmdet_encoder(cfgs.encoder, 3, frozen_bn=True, freeze_at=-1)
        self.imencoder_mt = build_mmdet_encoder(cfgs.encoder, 3, frozen_bn=True, freeze_at=-1)
        self.evencoder_tt = build_mmdet_encoder(cfgs.encoder, self.event_bins, frozen_bn=False, freeze_at=-1)
        self.evencoder_mt = build_mmdet_encoder(cfgs.encoder, self.event_bins, frozen_bn=False, freeze_at=-1)

        self.projector = nn.ModuleList([
            Fusion(cfgs.encoder_dim[i], self.projected_feature, 128 if cfgs.encoder == 'mobilenetv2' else 256, \
                   num_heads=self.att_heads[i]) for i in range(len(cfgs.encoder_dim))
        ])

        self.evmask_encoder = nn.Sequential(
            nn.Conv2d(1, self.hiddem_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hiddem_channels // 4),
            nn.ReLU(True),
            nn.Conv2d(self.hiddem_channels // 4, self.hiddem_channels // 2, kernel_size=3, stride=2, padding=1),
        )


        self.mask_decoder = MaskFormerHead(
            in_channels=[self.projected_feature] * 3,
            feat_channels=self.decoder_feat_channels,
            out_channels=self.decoder_out_channels,
            num_things_classes=self.max_label_num + 1,
            num_stuff_classes=0,
            num_queries=self.decoder_num_queries,
            pixel_decoder=dict(
                type='mmdet.PixelDecoder',
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU')),
            enforce_decoder_input_project=False,
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=self.decoder_feat_channels//2, normalize=True),
            transformer_decoder=dict(  # DetrTransformerDecoder
                return_intermediate=True,
                num_layers=6,
                layer_cfg=dict(  # DetrTransformerDecoderLayer
                    self_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=self.decoder_feat_channels,
                        num_heads=8,
                        attn_drop=0.1,
                        proj_drop=0.1,
                        dropout_layer=None,
                        batch_first=True),
                    cross_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=self.decoder_feat_channels,
                        num_heads=8,
                        attn_drop=0.1,
                        proj_drop=0.1,
                        dropout_layer=None,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=self.decoder_feat_channels,
                        feedforward_channels=2048,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.1,
                        dropout_layer=None,
                        add_identity=True)),
                init_cfg=None),
            loss_cls=self.loss_cls_dict,
            loss_mask=self.loss_focal_dict,
            loss_dice=self.loss_dice_dict,
            loss_bbox=self.loss_bbox_dict,
            train_cfg=dict(
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.ClassificationCost', weight=1.0),
                        dict(
                            type='mmdet.FocalLossCost',
                            weight=20.0,
                            binary_input=True),
                        dict(
                            type='mmdet.DiceCost',
                            weight=1.0,
                            pred_act=True,
                            eps=1.0)
                    ]),
                sampler=dict(type='mmdet.MaskPseudoSampler')),
        )

        self.mask_fusion = MaskFormerFusionHead(
            num_things_classes=self.max_label_num + 1,
            num_stuff_classes=0,
            test_cfg=dict(
                panoptic_on=False,
                semantic_on=False,
                instance_on=True,
                # max_per_image is for instance segmentation.
                max_per_image=self.max_obj_num,
                object_mask_thr=0.8,
                iou_thr=0.8,
                # In MaskFormer's panoptic postprocessing,
                # it will not filter masks whose score is smaller than 0.5 .
                filter_low_score=False),
        )

        self.flow_decoder = FPNHead(self.projected_feature, 2,
                                    shortcut_dims=[self.projected_feature] * 3,
                                    flow_branch_split=self.flow_branch_split)

        if not eval_mode:
            self.da_gamma = getattr(cfgs, "da_gamma", 10.0)
            self.da_lambda = 0.0
            self.total_steps = getattr(cfgs, "max_steps", 300000)
            self.current_step = 0
            self.texture_discriminator = DomainDiscriminator(cfgs.encoder_dim[-1])
            self.motion_discriminator = DomainDiscriminator(cfgs.encoder_dim[-1])
        else:
            self.da_gamma = None
            self.da_lambda = None
            self.total_steps = None
            self.current_step = None
            self.texture_discriminator = None
            self.motion_discriminator = None


        self.init_weights()

    def update_da_lambda(self):
        p = float(self.current_step) / float(self.total_steps)
        self.da_lambda = 2. / (1. + math.exp(-self.da_gamma * p)) - 1.

    def encoder_project(self, im_feats, ev_feats, ev_mask_feat=None):

        embs = []
        if len(im_feats) == 3:
            for i in range(len(im_feats)):
                x = self.projector[i](im_feats[i], ev_feats[i], ev_mask_feat)
                embs.append(x)
        elif len(im_feats) == 4: # for mobilenetv2 backbone
            for i in range(2):
                x = self.projector[i](im_feats[i], ev_feats[i], ev_mask_feat)
                embs.append(x)

            B, C, H, W = im_feats[3].shape
            im_feat3 = F.interpolate(im_feats[3].view(B, C // 4, H * 2, W * 2), \
                                     im_feats[2].shape[-2:], mode='bilinear', align_corners=True)
            ev_feat3 = F.interpolate(ev_feats[3].view(B, C // 4, H * 2, W * 2), \
                                     ev_feats[2].shape[-2:], mode='bilinear', align_corners=True)
            im_feat3 = torch.concat([im_feats[2], im_feat3], dim=1)
            ev_feat3 = torch.concat([ev_feats[2], ev_feat3], dim=1)
            x = self.projector[-1](im_feat3, ev_feat3, ev_mask_feat)
            embs.append(x)

        return embs

    def forward(self, inputs, is_Train=False):

        if is_Train:
            self.current_step += 1
            self.update_da_lambda()
        self.all_cls_scores_list = []
        self.all_mask_preds_list = []
        self.all_bbox_preds_list = []
        self.pred_flows_list = []
        self.mi_features_list = []
    
        self.inputs = inputs

        self.images = inputs['images']
        self.event_voxels = inputs['event_voxels']

        self.seq_length = min(self.inputs['seq_length'])
        self.pred_length = 1
        if is_Train and hasattr(self.cfgs, 'pred_length'):
            self.pred_length = self.cfgs['pred_length']
        self.batch_size = len(self.inputs['seq_length'])
        self.image_size = inputs['images'].shape[-2:]
        self.device = inputs['images'].device
        self.batch_img_metas = [{ # (height, width)
            'batch_input_shape': self.image_size,
            'img_shape': self.image_size,
            'ori_shape': self.image_size,
        } for _ in range(self.batch_size)]


        for seq_idx in range(self.seq_length):
            image1 = inputs['images'][:, seq_idx*3:(seq_idx+1)*3]
            event_voxel = inputs['event_voxels'][:, seq_idx*self.event_bins:(seq_idx+1)*self.event_bins]
            event_mask = None
            if hasattr(self, 'evmask_encoder'):
                event_mask = (torch.sum(event_voxel, dim=1, keepdim=True) != 0).float()

            embs, contrast_feats = self.encode(image1, event_voxel, event_mask)
            self.mi_features_list.append(contrast_feats)
            
            
            if seq_idx < self.pred_length:
                all_cls_score, all_mask_pred, all_bbox_preds = self.mask_decoder.forward(embs, self.batch_img_metas)
                self.all_cls_scores_list.append(all_cls_score)
                self.all_mask_preds_list.append(all_mask_pred)
                self.all_bbox_preds_list.append(all_bbox_preds)

                if hasattr(self.cfgs, 'eval_time') and self.cfgs.eval_time == True:
                    pred_flow = torch.zeros([image1.shape[0], 2, image1.shape[-2], image1.shape[-1]], \
                                                device=image1.device)
                else:
                    pred_flow = self.flow_decoder(embs[-1], embs) * 10
                    scale = [image1.shape[-2]/pred_flow.shape[-2], image1.shape[-1]/pred_flow.shape[-1]]
                    pred_flow = F.interpolate(pred_flow, scale_factor=scale, mode='bilinear', \
                        align_corners=True) * torch.tensor(scale, device=pred_flow.device).view(1, 2, 1, 1)

                self.pred_flows_list.append(pred_flow)

        if not is_Train:
            return self.output_fusion()
        else:
            return None
