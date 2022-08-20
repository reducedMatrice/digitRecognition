from __future__ import print_function
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torch.utils.model_zoo as model_zoo

train_losses = []
train_counter = []
test_losses = []
test_counter = []

# Adjust the model to get a higher performance
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 8, 5, 1) #input height, Convolution kernel, convolution kernal size, filter movement/step
        # self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels=16,kernel_size=5,stride=1) #input height
        self.conv2 = nn.Conv2d(16,32,5,1)
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(3200, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=6)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6)
        self.res_block_1 = ResidualBlock(32)
        self.res_block_2 = ResidualBlock(64)
        self.fc1 = nn.Linear(576, 10)

        # self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # self.res_block_1 = ResidualBlock(32)
        # self.res_block_2 = ResidualBlock(64)
        self.conv2_drop = nn.Dropout2d()
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.1)
        # self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.res_block_1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.res_block_2(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

#new model
class ResidualBlock(nn.Module):
    """
       每一个ResidualBlock,需要保证输入和输出的维度不变
       所以卷积核的通道数都设置成一样
       """

    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3,padding=1)

    def forward(self, x):
        """
        ResidualBlock中有跳跃连接;
        在得到第二次卷积结果时,需要加上该残差块的输入,
        再将结果进行激活,实现跳跃连接 ==> 可以避免梯度消失
        在求导时,因为有加上原始的输入x,所以梯度为: dy + 1,在1附近
        """
        y = F.gelu(self.conv1(x))
        y = self.conv2(y)

        return F.gelu(x + y)
    # 224*224
    # def __init__(self, block, num_layer, n_classes=1000, input_channels=3):
    #     super(Resnet, self).__init__()
    #     self.in_channels = 64
    #     self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #     self.bn1 = nn.BatchNorm2d(64)
    #     self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.layer1 = self._make_layer(block, 64, num_layer[0])
    #     self.layer2 = self._make_layer(block, 128, num_layer[1], 2)
    #     self.layer3 = self._make_layer(block, 256, num_layer[2], 2)
    #     self.layer4 = self._make_layer(block, 512, num_layer[3], 2)
    #     self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
    #     self.fc = nn.Linear(block.expansion * 512, n_classes)
    #
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1.0)
    #             nn.init.constant_(m.bias, 0.0)
    #
    # def _make_layer(self, block, out_channels, num_block, stride=1):
    #     downsample = None
    #     if stride != 1 or self.in_channels != out_channels * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
    #             nn.BatchNorm2d(out_channels * block.expansion)
    #         )
    #     layers = []
    #     layers.append(block(self.in_channels, out_channels, stride, downsample))
    #     self.in_channels = out_channels * block.expansion
    #     for _ in range(1, num_block):
    #         layers.append(block(self.in_channels, out_channels))
    #     return nn.Sequential(*layers)
    #
    # def forward(self, input):
    #     x = self.conv1(input)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     # x = self.layer4(x)
    #
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     return x

def train(args, model, device, train_loader, optimizer, epoch):
    global test_losses
    global test_counter
    global train_counter
    global train_losses
    if test_counter == []: test_counter = [i * len(train_loader.dataset) for i in range(epoch + 1)]
    model.train()
    #draw figure
    plt.figure()
    pic = None
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx in (1,2,3,4,5):
            if batch_idx == 1:
                pic = data[0,0,:,:]
            else:
                pic = torch.cat((pic,data[0,0,:,:]),dim=1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # Calculate gradients
        loss.backward()
        # Optimize the parameters according to the calculated gradients
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        # append this to the train_counter
        if batch_idx % 3 == 0:
            train_losses.append(math.log(loss.item()))
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

    plt.imshow(pic.cpu(), cmap='gray')



    # draw classification
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        predict = data[i][0]
        predict = predict.data.cpu().numpy()
        plt.imshow(predict, cmap='gray', interpolation='none')
        plt.title("Target VS prdiction: {} vs {}".format(target[i], output[i].argmax(dim=0, keepdim=True).item()),
                  fontdict={'family' : 'Times New Roman', 'size': 10})
        plt.xticks([])
        plt.yticks([])
    plt.show()




def test(model, device, test_loader):
    global  test_losses
    global test_counter
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(math.log(test_loss))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def draw_loss_graph(epochs):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue',linewidth=1)
    # plt.scatter(train_counter, train_losses, marker='o', s=16.0, color='blue')
    plt.plot(test_counter, test_losses, color='red')
    plt.scatter(test_counter, test_losses, marker='o', s=32.0, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 14)'),
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # batch_size is a crucial hyper-parameter
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        # Adjust num worker and pin memory according to your computer performance
        cuda_kwargs = {'num_workers': 3,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Normalize the input (black and white image)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Make train dataset split
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    # Make test dataset split
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    # Convert the dataset to dataloader, including train_kwargs and test_kwargs
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Put the model on the GPU or CPU
    # model = Net().to(device)
    model = Resnet().to(device)

    # Create optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Create a schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Begin training and testing
    test(model, device, test_loader)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        if epoch != 1: test_counter.append(epoch * len(train_loader.dataset))
        draw_loss_graph(args.epochs)

    # Save the model
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()