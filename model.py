import torch
import torch.nn as nn
from torch_module.utils import get_param_count
from torch_module.layers import Conv2D, DenseBlock, BottleNeckBlock


class NDenseNet(nn.Module):
    def __init__(self, growth_k=32):
        super(NDenseNet, self).__init__()
        self.growth_k = growth_k
        self.dense_count = [6, 12, 24, 16]
        self.__build__()

    def __build__(self):
        self.conv = Conv2D(3, self.growth_k*2, 7, 2, 3, 'relu', True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.dense_seq = nn.Sequential()
        self.transition_seq = nn.Sequential()
        input_ch = self.growth_k*2
        for j in range(4):
            dense_block = DenseBlock(input_ch, self.growth_k, self.dense_count[j])
            self.dense_seq.add_module(name='dense_{0}'.format(j+1), module=dense_block)
            output_ch = self.calc_k_output(input_ch, self.dense_count[j])

            if j != 3:
                transition_layer = nn.Sequential(
                    Conv2D(output_ch, output_ch//2, 1, 1, 0, 'relu', True),
                    nn.AvgPool2d(2, stride=2, padding=0)
                )
                self.transition_seq.add_module(name='tran_{0}'.format(j+1), module=transition_layer)
            input_ch = output_ch//2

        self.conv3 = Conv2D(output_ch, output_ch//2, 3, stride=2, padding=1, activation='relu')

        self.conv_3_1 = BottleNeckBlock(output_ch//2, output_ch//2)

        self.conv_3_2 = Conv2D(output_ch//2, output_ch//4, kernel_size=(3,4), stride=1,padding=0,activation='tanh')
        self.dense_3 = nn.Linear(output_ch//4, 63)

        self.conv_2_1 = BottleNeckBlock(output_ch//2, output_ch//2)

        self.conv_2_2 = Conv2D(output_ch//2, output_ch//4, kernel_size=(3,4), stride=1,padding=0,activation='tanh')
        self.dense_2 = nn.Linear(output_ch//4, 42)

    def calc_k_output(self, input_ch, k):
        return input_ch + k * self.growth_k

    def forward(self, x):

        # Dense Feature region
        x = self.conv(x)
        x = self.max_pool(x)
        for i in range(4):
            x = self.dense_seq[i](x)
            if i != 3:
                x = self.transition_seq[i](x)
        x = self.conv3(x)

        hand_3 = self.conv_3_1(x)
        hand_3 = self.conv_3_2(hand_3)
        hand_3 = hand_3.view((hand_3.shape[0], -1))
        hand_3 = self.dense_3(hand_3)
        hand_3 = hand_3.view((hand_3.shape[0], 21, 3))

        hand_2 = self.conv_2_1(x)
        hand_2 = self.conv_2_2(hand_2)
        hand_2 = hand_2.view((hand_2.shape[0], -1))
        hand_2 = self.dense_2(hand_2)
        hand_2 = hand_2.view((hand_2.shape[0], 21, 2))

        return hand_2, hand_3


if __name__ == "__main__":
    net = NDenseNet()
    print(get_param_count(net))
    t = torch.rand((1, 3, 192, 256))
    o = torch.rand((1,21,64,64))

    result = net(t)
    print(result.shape)

    optim = torch.optim.Adam(net.parameters(),lr=1e-4)
    criterion = torch.nn.MSELoss()

    for i in range(100):
        optim.zero_grad()
        result = net(t)
        print(torch.mean(torch.abs(result-o)))
        loss = criterion(o, result)
        loss.backward()
        optim.step()
        # for p in net.parameters():
        #     print(p)
        #     break
    print(result.shape)

