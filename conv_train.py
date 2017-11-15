"""
CNN for zener card classification.

:authors Jason, Nick, Sam
"""

import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import init_data

# Constants
RANDOM_SEED = 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)  # TODO change 10 to 4

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class ZenerDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        pass
    
    def __getitem__(self, img_path):
        img_data = rep_data(img_path)
        
        if transform:
            img_data = transform(img_data)

        return img_data


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # TODO replace with zener card transform logic
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


############################################################
#CLARGS
############################################################
parser = argparse.ArgumentParser(
    description='CNN for zener card classification',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

# Training settings
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

# Add positional CLARGS
# parser.add_argument(
#     'cost',
#     help='Specify the cost function [cross, cross-l1, cross-l2].'
# )
# parser.add_argument(
#     'network_description',
#     help='Path to network description file.'
# )
# parser.add_argument(
#     'epsilon',
#     type=float,
#     help='Epsilon error tolerance.'
# )
parser.add_argument(
    'max_updates',
    type=int,
    help='Number of training epochs.'
)
parser.add_argument(
    'class_letter',
    help='Specify the class letter [P, W, Q, S].'
)
parser.add_argument(
    'model_file_name',
    help='Filename to output trained model.'
)
parser.add_argument(
    'train_folder_name',
    help='Locating of training data.'
)


if __name__ == '__main__':
    args = parser.parse_args()

    # Init datasets
    input_data = init_data(args)  # dict of input data

    # CUDA setup
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(RANDOM_SEED)
    if args.cuda:
        torch.cuda.manual_seed(RANDOM_SEED)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    model = Net()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0  # L2 penalty value
    )

    # Data import
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Train & test per epoch
    for epoch in range(1, args.max_updates + 1):
        train(epoch)
        test()
