# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------


import torch.nn as nn
import torchvision.models as models

BN_MOMENTUM = 0.1


class MobiletNetHM(nn.Module):
    def __init__(self, heads, head_conv, **kwargs):
        self.inplanes = 576
        self.deconv_with_bias = False
        self.heads = heads

        super().__init__()
        self.features = models.mobilenet_v3_small(pretrained=True).features

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )
        # self.final_layer = []

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        head_conv,
                        num_output,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )
            else:
                fc = nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            self.__setattr__(head, fc)

        # self.final_layer = nn.ModuleList(self.final_layer)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(
            num_filters
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"
        assert num_layers == len(
            num_kernels
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias,
                )
            )
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret

    def init_weights(self):
        # print('=> init resnet deconv weights from normal distribution')
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # print('=> init final conv weights from normal distribution')
        for head in self.heads:
            final_layer = self.__getattr__(head)
            for i, m in enumerate(final_layer.modules()):
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    if m.weight.shape[0] == self.heads[head]:
                        if "hm" in head:
                            nn.init.constant_(m.bias, -2.19)
                        else:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)


def get_mobilenet_v3_small(num_layers, heads, head_conv):

    model = MobiletNetHM(heads, head_conv=head_conv)
    model.init_weights()
    return model
