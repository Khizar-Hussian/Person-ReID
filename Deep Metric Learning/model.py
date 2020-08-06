import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import TrainLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from Binomial import BinomialDevianceLoss


class SCNN(nn.Module):
    def __init__(self):
        """
            padding is calculated as: Padding = (Kernel_size-1) / 2
            Kernel Sizes were predefined in the paper
        """
        super().__init__()
        self.C1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.C31 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.C32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.C33 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.F5 = nn.Linear(64*12*12, 500)

    def forward_one(self, x):
        x1 = torch.narrow(x, 2, 0, 48)
        x2 = torch.narrow(x, 2, 40, 48)
        x3 = torch.narrow(x, 2, 80, 48)
        
        x1 = F.max_pool2d(F.relu(self.C1(x1)), (2,2))
        x2 = F.max_pool2d(F.relu(self.C1(x2)), (2,2))
        x3 = F.max_pool2d(F.relu(self.C1(x3)), (2,2))

        x1 = F.max_pool2d(F.relu(self.C31(x1)), (2,2))
        x2 = F.max_pool2d(F.relu(self.C32(x2)), (2,2))
        x3 = F.max_pool2d(F.relu(self.C33(x3)), (2,2))

        X = x1+x2+x3
        X = X.view(-1, 64*12*12)
        X = F.relu(self.F5(X))
        return X

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2


def generate_masks(x1, x2, labels, c):
    M = torch.zeros((x1.shape[0], x2.shape[0]))

    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            if j == i:
                M[i][j] = labels[i]
            elif j > i:
                M[i][j] = -c

    n1 = (M == 1).sum().item()
    n2 = (M == -c).sum().item()
    W = M.detach().clone()
    # W = torch.where(W==1 & W!=0, torch.tensor(1/n1), torch.tensor(1/n2))
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if W[i][j] == 1:
                W[i][j] = 1/n1
            elif W[i][j] == -1:
                W[i][j] = 1/n2
    return M, W    



def main():
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    training_set = TrainLoader(os.path.normpath('D:\\Academia\\Semester 7\\FYP Papers\\Datasets\\prid_2011\\single_shot\\train'), dataset_name='PRID', transform=data_transforms)
    generator = DataLoader(training_set, 128, True)
    model = SCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for batch1, batch2, labels in generator:
        optimizer.zero_grad()
        x, y = model(batch1, batch2)
        M, W = generate_masks(x, y, labels, 1)
        criterion = BinomialDevianceLoss()
        loss = criterion(x, y, M, W)
        print(f"loss: {loss}")
        loss.backward()
        optimizer.step()    

if __name__ == "__main__":
    main()
    