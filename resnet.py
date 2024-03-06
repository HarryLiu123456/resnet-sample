import torch
import torch.nn as nn

# inchannel -> outchannel
class BasicBlock(nn.Module):
    #判断残差结构中，主分支的卷积核个数是否发生变化，不变则为1
    expansion = 1
 
    #downsample是为了确保最后相加的两个张量形状一样
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        #因为核函数是3*3，所以需要padding一圈，以维持大小不变
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        #中间值批量归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.downsample = downsample
 
    def forward(self, x):
        #残差块保留原始输入
        identity = x
        #根据情况选择是否下采样
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
 
        return out

class ResNet(nn.Module):
    #block接受一个类，blocks_num接受一个列表，include_top指是否有最后的全连接层
    #groups指定分组卷积的组数，width_per_group指定每个组的通道数
    #对于basicblock，groups和width没有用，在bottleneck中有用
    def __init__(self, block, blocks_num, num_classes=100, include_top=True, groups=1, width_per_group=64):
        super(ResNet, self).__init__()

        self.include_top = include_top
        #maxpool的输出通道数为64，残差结构输入通道数为64
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        # 3->in_channel
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        # 这里我先改成False，以免出错
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 浅层的stride=1，深层的stride=2
        # block：定义的两种残差模块
        # block_num：模块中残差块的个数
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            # 自适应平均池化，指定输出（H，W），通道数不变
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # 全连接层
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 这里是进行训练的初始化参数
        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # kaiming正态分布初始化，使得Conv2d卷积层反向传播的输出的方差都为1
                # fan_in：权重是通过线性层（卷积或全连接）隐性确定
                # fan_out：通过创建随机矩阵显式创建权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 
    # 定义残差模块，由若干个残差块组成
    # block：定义的两种残差模块，channel：该模块中所有卷积层的基准通道数。block_num：模块中残差块的个数
    def _make_layer(self, block, channel, block_num, stride=1):

        downsample = None

        # 如果满足条件，则是虚线残差结构
        if stride != 1 or self.in_channel != channel * block.expansion:
            # nn.sequential就是把构建过程简单化了
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
 
        layers = []
        # 如果使用basicblock则groups和width无用
        #第一层，如果进行下采样也只有第一层下采样
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, 
                            groups=self.groups, width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
 
        #其他层
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups,
                            width_per_group=self.width_per_group))
            
        # Sequential：自定义顺序连接成模型，生成网络结构
        return nn.Sequential(*layers)
 
    def forward(self, x):

        # 无论哪种ResNet，都需要的静态层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 动态层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        if self.include_top:
            x = self.avgpool(x)
            # 将张量铺平
            x = torch.flatten(x, 1)
            # 过一个线性层
            x = self.fc(x)
 
        return x
    
def resnet34(num_classes=100, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

