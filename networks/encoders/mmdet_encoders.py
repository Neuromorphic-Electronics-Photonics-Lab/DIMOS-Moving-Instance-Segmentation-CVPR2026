from mmdet.models.backbones.resnet import ResNet
from mmdet.models.backbones.mobilenet_v2 import MobileNetV2
from mmengine.dist import master_only


class MobileNetV2_modified(MobileNetV2):
    def __init__(self, input_dim=3, **other_params):
        super().__init__(**other_params)

        from mmcv.cnn import ConvModule
        self.conv1 = ConvModule(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        del self._modules['conv2']
        self.layers.pop()

    @master_only
    def _dump_init_info(self):
        pass


class ResNet_modified(ResNet):
    
    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                for _ in range(self.out_indices.count(i)):
                    outs.append(x)
        return tuple(outs)

    @master_only
    def _dump_init_info(self):
        pass


def build_mmdet_encoder(name, input_dim=3, frozen_bn=True, freeze_at=-1):
    if name == 'mobilenetv2':
        return MobileNetV2_modified(input_dim=input_dim,
                                    out_indices=(1, 2, 4, 6),
                                    frozen_stages=freeze_at,
                                    norm_cfg=dict(
                                        type='BN', requires_grad=not frozen_bn),
                                    norm_eval=frozen_bn,
                                    init_cfg=dict(
                                        type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2'))
    elif name == 'resnet50':
        return ResNet_modified(in_channels=input_dim,
                               depth=50,
                               num_stages=3,
                               strides=(1, 2, 2),
                               dilations=(1, 1, 1),
                               out_indices=(0, 1, 2),
                               frozen_stages=freeze_at,
                               norm_cfg=dict(
                                   type='BN', requires_grad=not frozen_bn),
                               norm_eval=frozen_bn,
                               style='pytorch',
                               init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
    elif name == 'resnet50_dcn':
        return ResNet_modified(in_channels=input_dim,
                               depth=50,
                               num_stages=3,
                               strides=(1, 2, 2),
                               dilations=(1, 1, 1),
                               out_indices=(0, 1, 2),
                               frozen_stages=freeze_at,
                               norm_cfg=dict(
                                   type='BN', requires_grad=not frozen_bn),
                               norm_eval=frozen_bn,
                               style='pytorch',
                               init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
                               dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
                               stage_with_dcn=(False, True, True))
    else:
        raise NotImplementedError
