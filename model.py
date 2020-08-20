import torch
import torch.nn as nn
from torch_module.utils import get_param_count
from torch_module.layers import Conv2D, DenseBlock, BottleNeckBlock, UpConv2D


class DenseUNetFilter(nn.Module):
    def __init__(self, growth_k=16, activation='relu', using_up=False, using_down=False):
        super(DenseUNetFilter, self).__init__()
        self.growth_k = growth_k
        self.dense_count = [4, 6, 12]
        self.activation = activation
        self.using_down = using_down
        self.using_up = using_up

        self.__build__()

    def __build__(self):
        self.conv = Conv2D(3, self.growth_k*2, 7, 2, 3, self.activation, True)
        self.conv3 = Conv2D(self.growth_k*2, self.growth_k*2, 3, 1, 1, activation=self.activation, batch=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.down_dense_list = nn.ModuleList()
        self.down_pool_list = nn.ModuleList()

        self.up_dense_list = nn.ModuleList()
        self.up_pool_list = nn.ModuleList()

        self.base_connection = nn.ModuleList()
        self.down_connect_list = nn.ModuleList()
        self.up_connect_list = nn.ModuleList()

        model_depth = len(self.dense_count)
        input_ch_list = [self.growth_k*2]
        # input_ch = self.growth_k*2

        ##########################################################################
        # Down sampling Region
        ##########################################################################

        for j in range(model_depth):
            input_ch = input_ch_list[-1]
            dense_block = DenseBlock(input_ch, self.growth_k, self.dense_count[j])
            self.down_dense_list.append(dense_block)

            output_ch = dense_block.out_ch

            down_pool_layer = nn.Sequential(
                Conv2D(output_ch, output_ch, 1, 1, 0, self.activation,True),
                nn.AvgPool2d(2, stride=2, padding=0)
            )
            self.down_pool_list.append(down_pool_layer)

            input_ch_list.append(output_ch)

            if j != len(self.dense_count)-1 and self.using_down:
                self.down_connect_list.append(nn.Sequential(
                    nn.AvgPool2d(2,2,0),
                    BottleNeckBlock(output_ch, True, activation=self.activation)
                ))

            if j != 0 and self.using_up:
                self.up_connect_list.append(nn.Sequential(
                    nn.Upsample(scale_factor=2,mode='nearest'),
                    BottleNeckBlock(output_ch, True, activation=self.activation)
                ))

            self.base_connection.append(BottleNeckBlock(output_ch, True, activation=self.activation))
        input_ch = input_ch_list[-1]
        self.middle_block = nn.Sequential(
            BottleNeckBlock(input_ch, True, activation=self.activation),
            BottleNeckBlock(input_ch, True, activation=self.activation),
        )

        print(input_ch_list)

        ##########################################################################
        # Up sampling Region
        ##########################################################################
        for j in range(model_depth):
            up_pool_layer = UpConv2D(2, input_ch, input_ch, 3, 1, 1, activation=self.activation, batch=True)
            self.up_pool_list.append(up_pool_layer)
            print(j, input_ch)
            input_ch = input_ch + input_ch_list[-j-1]
            if j != model_depth-1 and self.using_down:
                input_ch = input_ch + input_ch_list[-j-2]
            if j != 0 and self.using_up:
                input_ch = input_ch + input_ch_list[-j]
            print(j, input_ch)

            dense_block = DenseBlock(input_ch, self.growth_k, self.dense_count[len(self.dense_count) - j - 1])
            self.up_dense_list.append(nn.Sequential(
                dense_block,
                Conv2D(dense_block.out_ch, input_ch_list[-j-2], 3, 1, 1, activation=self.activation, batch=True)
            ))
            input_ch = input_ch_list[-j-2]
        self.joint_2d_b = BottleNeckBlock(input_ch, attention=True, activation=self.activation)
        self.joint_2d = Conv2D(input_ch, 21, 1, 1, 0, 'sigmoid')

        self.joint_3d_b = BottleNeckBlock(input_ch, attention=True, activation=self.activation)
        self.joint_3d_l = nn.Sequential(
            nn.Linear(input_ch, 128),
            nn.BatchNorm1d(128)
        )
        self.joint_3d = nn.Linear(128, 60)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(self.conv3(x))

        model_depth = len(self.dense_count)
        down_connect = []
        up_connect = []
        base_connect = []
        # Down sampling
        for j in range(model_depth):
            x = self.down_dense_list[j](x)
            base = self.base_connection[j](x)
            base_connect.append(base)
            if j != model_depth-1 and self.using_down:
                down = self.down_connect_list[j](x)
                down_connect.append(down)

            if j != 0 and self.using_up:
                up = self.up_connect_list[j-1](x)
                up_connect.append(up)
            x = self.down_pool_list[j](x)
        x = self.middle_block(x)

        down_count = -1
        up_count = -1
        for j in range(model_depth):
            up_base = self.up_pool_list[j](x)
            connect = torch.cat((up_base, base_connect[model_depth - j - 1]),dim=1)
            if j != model_depth-1 and self.using_down:
                connect = torch.cat((connect, down_connect[down_count]), dim=1)
                down_count = down_count-1
            if j != 0 and self.using_up:
                connect = torch.cat((connect, up_connect[up_count]), dim=1)
                up_count = up_count-1
            x = self.up_dense_list[j](connect)

        print(down_count, up_count, len(down_connect))

        joint_2d = self.joint_2d_b(x)
        joint_2d = self.joint_2d(joint_2d)

        joint_3d = self.joint_3d_b(x)
        joint_3d = torch.mean(joint_3d, dim=[2, 3])
        joint_3d = joint_3d.view((joint_3d.shape[0], -1))
        joint_3d = torch.relu(self.joint_3d_l(joint_3d))
        joint_3d = torch.tanh(self.joint_3d(joint_3d))

        return joint_2d, joint_3d


class DenseUNet(nn.Module):
    def __init__(self, growth_k=16, activation='relu', using_up=False, using_down=False):
        super(DenseUNet, self).__init__()
        self.growth_k = growth_k
        self.dense_count = [4, 4, 8]
        self.activation = activation

        self.__build__()

    def __build__(self):
        self.conv = Conv2D(3, self.growth_k*2, 7, 2, 3, self.activation, True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.down_dense_list = nn.ModuleList()
        self.down_pool_list = nn.ModuleList()
        input_ch = self.growth_k*2
        depth_ch = []
        for j in range(len(self.dense_count)):
            dense_block = DenseBlock(input_ch, self.growth_k, self.dense_count[j])
            self.down_dense_list.append(dense_block)
            output_ch = self.calc_k_output(input_ch, self.dense_count[j])
            depth_ch.append(output_ch)

            if j != len(self.dense_count)-1:
                down_pool_layer = nn.Sequential(
                    Conv2D(output_ch, output_ch, 1, 1, 0, self.activation,True),
                    nn.AvgPool2d(2, stride=2, padding=0)
                )
                self.down_pool_list.append(down_pool_layer)
            input_ch = output_ch

        self.middle_bottle_list = nn.ModuleList()
        for j in range(len(self.dense_count)):
            self.middle_bottle_list.append(BottleNeckBlock(depth_ch[j], attention=True, activation=self.activation))

        self.middle_expand_list = nn.ModuleList()
        for depth_idx in range(1, len(self.dense_count)):
            up_module = nn.ModuleList()
            for up_idx in range(depth_idx):
                up_seq = nn.Sequential()
                for u in range(up_idx+1):
                    up_conv = UpConv2D(2, depth_ch[depth_idx], depth_ch[depth_idx], 3, 1, 1, self.activation, True)
                    up_seq.add_module(name='skip_up_connection_{0}_{1}'.format(up_idx, u), module=up_conv)
                up_seq.add_module(name='skip_conv_1_{0}'.format(up_idx),
                                  module=Conv2D(depth_ch[depth_idx], depth_ch[depth_idx-(up_idx+1)],
                                                1, 1, 0, self.activation, True))
                up_module.append(up_seq)

            self.middle_expand_list.append(up_module)

        self.up_dense_list = nn.ModuleList()
        self.up_pool_list = nn.ModuleList()

        for j in range(len(self.dense_count)):
            dense_block = DenseBlock(depth_ch[j], self.growth_k, self.dense_count[j])
            self.up_dense_list.append(dense_block)

            if j != 0:
                self.up_pool_list.append(UpConv2D(2, self.calc_k_output(depth_ch[j], self.dense_count[j]), depth_ch[j-1], 1, 1, 0, activation=self.activation, batch=True))

        last_ch = self.calc_k_output(depth_ch[0], self.dense_count[0])
        self.joint_2d_b = BottleNeckBlock(last_ch, attention=True, activation=self.activation)
        self.joint_2d = Conv2D(last_ch, 21, 1, 1, 0, 'sigmoid')

        self.joint_3d_b = BottleNeckBlock(last_ch, attention=True, activation=self.activation)
        self.joint_3d_l = nn.Sequential(
            nn.Linear(last_ch, 128),
            nn.BatchNorm1d(128)
        )
        self.joint_3d = nn.Linear(128, 60)

    def calc_k_output(self, input_ch, k):
        return input_ch + k * self.growth_k

    def forward(self, x):

        # Dense Feature region
        x = self.conv(x)
        x = self.max_pool(x)

        middle_output = []

        for i in range(len(self.dense_count)):
            x = self.down_dense_list[i](x)
            middle_output.append(x)
            if i != len(self.dense_count)-1:
                x = self.down_pool_list[i](x)

        for i in range(len(self.dense_count)):
            next_out = middle_output[i]
            for k in range(i, len(self.middle_expand_list)):
                up_list = self.middle_expand_list[k]
                up_seq = up_list[-(i+1)]
                up_seq_result = up_seq(middle_output[k+1])
                next_out = next_out + up_seq_result
            middle_output[i] = next_out

        for i in range(len(self.dense_count)):
            middle_output[i] = self.middle_bottle_list[i](middle_output[i])

        for i in range(len(self.dense_count)-1, -1, -1):
            if i == len(self.dense_count)-1:
                up_dense = self.up_dense_list[i](middle_output[i])
                up_pool = self.up_pool_list[i-1](up_dense)
            elif i != 0:
                up_dense = self.up_dense_list[i](up_pool+middle_output[i])
                up_pool = self.up_pool_list[i-1](up_dense)
            else:
                up_dense = self.up_dense_list[i](up_pool+middle_output[i])

        joint_2d = self.joint_2d_b(up_dense)
        joint_2d = self.joint_2d(joint_2d)

        joint_3d = self.joint_3d_b(up_dense)
        joint_3d = torch.mean(joint_3d, dim=[2, 3])
        joint_3d = joint_3d.view((joint_3d.shape[0], -1))
        joint_3d = torch.relu(self.joint_3d_l(joint_3d))
        joint_3d = torch.tanh(self.joint_3d(joint_3d))

        return joint_2d, joint_3d


if __name__ == "__main__":
    net = DenseUNetFilter(using_down=True, using_up=True)
    print('{:,}'.format(get_param_count(net)))
    t = torch.rand((4, 3, 192, 256))
    o = torch.rand((4,21,64,64))

    result = net(t)

    optim = torch.optim.Adam(net.parameters(),lr=1e-4)
    criterion = torch.nn.MSELoss()

    print(result[0].shape)
    print(result[1].shape)

