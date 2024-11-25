import torch
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict
from torch.nn import BatchNorm3d as bn


class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16  //output_stride除和乘的数
    """
    def __init__(self, model_cfg, n_class=19, output_stride=8):
        super(DenseASPP, self).__init__()
        bn_size = model_cfg['bn_size']
        drop_rate = model_cfg['drop_rate']
        growth_rate = model_cfg['growth_rate']
        num_init_features = model_cfg['num_init_features']
        block_config = model_cfg['block_config']

        dropout0 = model_cfg['dropout0']
        dropout1 = model_cfg['dropout1']
        d_feature0 = model_cfg['d_feature0']
        d_feature1 = model_cfg['d_feature1']

        feature_size = int(output_stride / 8)  #这个和后面乘回来的是一个数

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(4, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', bn(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=1, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        # block1*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 1, block)
        num_features = num_features + block_config[0] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 1, trans)
        num_features = num_features // 2

        # block2*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 2, block)
        num_features = num_features + block_config[1] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=feature_size)
        self.features.add_module('transition%d' % 2, trans)
        num_features = num_features // 2

        # block3*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(2 / feature_size))
        self.features.add_module('denseblock%d' % 3, block)
        num_features = num_features + block_config[2] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 3, trans)
        num_features = num_features // 2

        # block4*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(4 / feature_size))
        self.features.add_module('denseblock%d' % 4, block)
        num_features = num_features + block_config[3] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 4, trans)
        num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', bn(num_features))
        if feature_size > 1:
            self.features.add_module('upsample', nn.Upsample(scale_factor=2, mode='trilinear'))

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)
        num_features = num_features + 5 * d_feature1

        self.classification = nn.Sequential(
            nn.Dropout3d(p=dropout1),
            nn.Conv3d(in_channels=num_features, out_channels=n_class, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=4, mode='trilinear'),
        )

        # self.out = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(4, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    # @profile
    def forward(self, _input): #[1 4 32 96 96]
        feature = self.features(_input) #  64*32 [1 18 8 24 24] [1 5 4 12 12]   4-8-1

        aspp3 = self.ASPP_3(feature) #[1 4 8 24 24]
        feature = torch.cat((aspp3, feature), dim=1) #[1 27 4 12 12]

        aspp6 = self.ASPP_6(feature) #[1 8 8 24 24]
        feature = torch.cat((aspp6, feature), dim=1) #[1 35 4 12 12]

        aspp12 = self.ASPP_12(feature) #[1 8 8 24 24]
        feature = torch.cat((aspp12, feature), dim=1)#[1 43 4 12 12]

        aspp18 = self.ASPP_18(feature) #[1 64 8 24 24]
        feature = torch.cat((aspp18, feature), dim=1)#[1 294 4 12 12]

        aspp24 = self.ASPP_24(feature) #[1 64 8 24 24]
        feature = torch.cat((aspp24, feature), dim=1) ##[1 38 4 12 12]

        cls = self.classification(feature) #[1 4 8 24 24]
        # cls = self.out(cls)

        return cls #[1 4 32 96 96]
    # def forward(self, _input): 
    #     feature = self.features(_input)  # [1, 18, 8, 24, 24]

    #     aspp3 = self.ASPP_3(feature)  # [1, 4, 8, 24, 24]
    #     feature = torch.cat((aspp3, feature), dim=1)  # [1, 27, 4, 12, 12]
    #     del aspp3  # 删除中间变量，释放内存

    #     aspp6 = self.ASPP_6(feature)  # [1, 8, 8, 24, 24]
    #     feature = torch.cat((aspp6, feature), dim=1)  # [1, 35, 4, 12, 12]
    #     del aspp6  # 删除中间变量，释放内存

    #     aspp12 = self.ASPP_12(feature)  # [1, 8, 8, 24, 24]
    #     feature = torch.cat((aspp12, feature), dim=1)  # [1, 43, 4, 12, 12]
    #     del aspp12  # 删除中间变量，释放内存

    #     aspp18 = self.ASPP_18(feature)  # [1, 64, 8, 24, 24]
    #     feature = torch.cat((aspp18, feature), dim=1)  # [1, 294, 4, 12, 12]
    #     del aspp18  # 删除中间变量，释放内存

    #     aspp24 = self.ASPP_24(feature)  # [1, 64, 8, 24, 24]
    #     feature = torch.cat((aspp24, feature), dim=1)  # [1, 38, 4, 12, 12]
    #     del aspp24  # 删除中间变量，释放内存

    #     cls = self.classification(feature)  # [1, 4, 8, 24, 24]

    #     return cls


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm_1', bn(input_num, momentum=0.0003)),

        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv3d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm_2', bn(num1, momentum=0.0003)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv3d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)

        return feature


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation_rate=1):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', bn(num_input_features)),
        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm_2', bn(bn_size * growth_rate)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, dilation=dilation_rate, padding=dilation_rate, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation_rate=1):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, dilation_rate=dilation_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', bn(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=stride))


if __name__ == "__main__":
    model = DenseASPP(2)
    print(model)
