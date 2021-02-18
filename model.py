
import torch.nn as nn
import torch.nn.functional as F
import torch



class Net(nn.Module):

    def Layer(self, layer_count, channels, channels_in, stride, kernal=[45, 7], padding=[23, 3]):
        return nn.Sequential(
            Block(channels_in, channels, stride, kernal=kernal, padding=padding),
            *[Block(channels, channels, kernal=kernal, padding=padding) for _ in range(layer_count-1)])

    def __init__(self, n=4):
        super(Net, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        self.pre1 = self.Layer(n, channels=6, channels_in=1, stride=1, kernal=[5, 7], padding=[2, 3])
        self.pre2 = self.Layer(n, channels=16, channels_in=6, stride=1, kernal=[11, 7], padding=[5, 3])
        self.pre3 = self.Layer(n, channels=32, channels_in=16, stride=1, kernal=[45, 7], padding=[22, 3])
        self.conv1 = self.Layer(n, channels=64, channels_in=32, stride=2, kernal=[45, 7], padding=[22, 3])
        self.conv2 = self.Layer(n, channels=128, channels_in=64, stride=2, kernal=[45, 7], padding=[22, 3])
        self.conv3 = self.Layer(n, channels=256, channels_in=128, stride=2, kernal=[45, 7], padding=[22, 3])
        # self.conv3 = self.Layer(n, channels=512, channels_in=256, stride=1, kernal=[45, 7], padding=[22, 3])
        self.pool = nn.MaxPool2d([6, 4])
        self.linear = nn.Linear(256, 1)



    



    def forward(self, x):
        guide = x[0].to(self.device)
        target = x[1].to(self.device)
        guide = F.relu(self.pre1(guide))
        guide = F.relu(self.pre2(guide))
        guide = F.relu(self.pre3(guide))
        target = F.relu(self.pre1(target))


        target = F.relu(self.pre2(target))
        target = F.relu(self.pre3(target))

        out = torch.cat((guide, target), dim=2)
        # print(out.size())
        # print(out.size())
        out = self.conv1(out)

        out = self.conv2(out)
        # print(out.size())
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        # print(out.size())
        # print(1)
        try:
            out = self.linear(out)
        except RuntimeError:
            print(x[1].size())
        return F.relu(out)




class Block(nn.Module):

    def __init__(self, channels_in, num_filters, stride=1, kernal=[45, 7], padding=[22, 3]):
        super(Block, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else:
            self.projection = IdentityPadding(num_filters, channels_in, stride)
        self.to("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=kernal, stride=[stride, 1], padding=padding)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernal, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(.01)

    def forward(self, x):
        oldx = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.projection:
            oldx = self.projection(oldx)
        x += oldx
        x = self.relu2(x)
        return x




class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.i = nn.MaxPool2d(1, stride=[stride, 1])
        self.num_zeros = num_filters - channels_in
        self.to("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.i(out)
        return out


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.layer = nn.Linear(184, 1)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x):
        guide = x[0].to(self.device)
        target = x[1].to(self.device)
        out = torch.cat((guide, target), dim=2)
        out = out.view(out.size(0), -1)
        out = self.layer(out)
        return out

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.layer = nn.Linear(184, 1)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x):
        guide = x[0].to(self.device)
        target = x[1].to(self.device)
        out = torch.cat((guide, target), dim=2)
        out = out.view(out.size(0), -1)
        out = F.sigmoid(self.layer(out))
        return out

