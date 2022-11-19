import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
from MRWA_Net import MrwaNet
from noise import add_noise


device = torch.device('cuda')

data = np.load('data_multi_label.npz')
trainx = data['arr_0']  # (38015, 52. 52)
# trainx = add_noise(trainx, 4)
trainy = data['arr_1']  # (38015, 8), 0-1999: C24, 34866-35014: C9, 33000-33865: C7
x_train, x_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.2, random_state=10)
BATCH_SIZE = 128


# Load model
class TrainSet(Dataset):
    def __init__(self):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        train_data = torch.tensor(self.x_train[index])
        train_label = torch.tensor(self.y_train[index])
        return train_data, train_label

    def __len__(self):
        return self.x_train.shape[0]

class TestSet(Dataset):
    def __init__(self):
        self.x_test = x_test
        self.y_test = y_test

    def __getitem__(self, index):
        test_data = torch.tensor(self.x_test[index])
        test_label = torch.tensor(self.y_test[index])
        return test_data, test_label

    def __len__(self):
        return self.x_test.shape[0]

set1 = TrainSet()
set2 = TestSet()
train_loader = DataLoader(dataset=set1, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=set2, batch_size=BATCH_SIZE, shuffle=True)

lr = 0.001

EPOCH = 50
best_acc = 0
model = MrwaNet().to(device)
dictx = {'[0, 0, 0, 0, 0, 0, 0, 0]': 0,
        '[1, 0, 0, 0, 0, 0, 0, 0]': 1,
        '[0, 1, 0, 0, 0, 0, 0, 0]': 2,
        '[0, 0, 1, 0, 0, 0, 0, 0]': 3,
        '[0, 0, 0, 1, 0, 0, 0, 0]': 4,
        '[0, 0, 0, 0, 1, 0, 0, 0]': 5,
        '[0, 0, 0, 0, 0, 1, 0, 0]': 6,
        '[0, 0, 0, 0, 0, 0, 1, 0]': 7,
        '[0, 0, 0, 0, 0, 0, 0, 1]': 8,
        '[1, 0, 1, 0, 0, 0, 0, 0]': 9,
        '[1, 0, 0, 1, 0, 0, 0, 0]': 10,
        '[1, 0, 0, 0, 1, 0, 0, 0]': 11,
        '[1, 0, 0, 0, 0, 0, 1, 0]': 12,
        '[0, 1, 1, 0, 0, 0, 0, 0]': 13,
        '[0, 1, 0, 1, 0, 0, 0, 0]': 14,
        '[0, 1, 0, 0, 1, 0, 0, 0]': 15,
        '[0, 1, 0, 0, 0, 0, 1, 0]': 16,
        '[0, 0, 1, 0, 1, 0, 0, 0]': 17,
        '[0, 0, 1, 0, 0, 0, 1, 0]': 18,
        '[0, 0, 0, 1, 1, 0, 0, 0]': 19,
        '[0, 0, 0, 1, 0, 0, 1, 0]': 20,
        '[0, 0, 0, 0, 1, 0, 1, 0]': 21,
        '[1, 0, 1, 0, 1, 0, 0, 0]': 22,
        '[1, 0, 1, 0, 0, 0, 1, 0]': 23,
        '[1, 0, 0, 1, 1, 0, 0, 0]': 24,
        '[1, 0, 0, 1, 0, 0, 1, 0]': 25,
        '[1, 0, 0, 0, 1, 0, 1, 0]': 26,
        '[0, 1, 1, 0, 1, 0, 0, 0]': 27,
        '[0, 1, 1, 0, 0, 0, 1, 0]': 28,
        '[0, 1, 0, 1, 1, 0, 0, 0]': 29,
        '[0, 1, 0, 1, 0, 0, 1, 0]': 30,
        '[0, 1, 0, 0, 1, 0, 1, 0]': 31,
        '[0, 0, 1, 0, 1, 0, 1, 0]': 32,
        '[0, 0, 0, 1, 1, 0, 1, 0]': 33,
        '[1, 0, 1, 0, 1, 0, 1, 0]': 34,
        '[1, 0, 0, 1, 1, 0, 1, 0]': 35,
        '[0, 1, 1, 0, 1, 0, 1, 0]': 36,
        '[0, 1, 0, 1, 1, 0, 1, 0]': 37
        }
array = np.zeros((39, 39), dtype=int)
list = [[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0]]
train_loss = []
test_loss = []


def threshold_loss(Lcard, t, output, total, sum):
    sum += (output > t).sum()
    return Lcard - sum / total, sum


def save_model(epoch, optimizer, average_loss, pred_acc):
    '''if not os.path.isdir('baseline'):
        os.mkdir('baseline')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': round(np.mean(average_loss), 2)
    }, '/home/baseline' + f'/baseline0.001_Epoch-{epoch}-Test_loss-{round(np.mean(average_loss), 4)}-{pred_acc*100}%.pth')'''
    print(f"\nBest accuracy:{pred_acc*100}%")


def test(criterion, epoch, optimizer):
    global best_acc
    threshold = 0.5
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    pbar = tqdm(test_loader, desc=f'Test Epoch{epoch}/{EPOCH}')
    model.eval()
    with torch.no_grad():
        with pbar as t:
            for data, label in t:
                batch_size = label.shape[0]
                data, label = data.float().unsqueeze(1).cuda(), label.float()
                output = model(data)

                output = output.cpu()
                output = torch.sigmoid(output)
                if not torch.isnan(output).any():
                    loss = criterion(output, label).item()
                    test_loss.append(loss)

                    pred = torch.where(output > threshold, torch.ones_like(output), torch.zeros_like(output))
                    # correct += (pred == label).sum()
                    total += label.shape[0]
                    i = 0
                    while i < label.shape[0]:
                        if all(label[i] == pred[i]):
                            correct += 1

                        if f'{pred[i].cpu().numpy().astype(int).tolist()}' in dictx:
                            array[dictx[f'{label[i].cpu().numpy().astype(int).tolist()}']][
                                dictx[f'{pred[i].cpu().numpy().astype(int).tolist()}']] += 1
                        else:
                            array[dictx[f'{label[i].cpu().numpy().astype(int).tolist()}']][38] += 1
                        i += 1

                pbar.set_description(
                    f'Test  Epoch: {epoch}/{EPOCH} ')
            pred_acc = correct / total
            print("Test Accuracy of the epoch: ", pred_acc*100)
    if pred_acc > best_acc:
        best_acc = pred_acc
        save_model(epoch, optimizer, test_loss, pred_acc)


def train():
    label_num = 0
    total = 0
    sum = 0
    threshold = 0.5
    criterion = F.binary_cross_entropy
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    # lambda1 = lambda epoch: 0.5 ** (epoch // 10)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    for epoch in range(EPOCH):
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}/{EPOCH}')
        model.train()
        with pbar as t:
            for batch_idx, (data, label) in enumerate(t):
                total += label.shape[0] * label.shape[1]

                data, label = data.unsqueeze(1).float().cuda(), label.float().cuda()
                optimizer.zero_grad()
                output = model(data)
                output = torch.sigmoid(output)

                loss = criterion(output, label)
                # loss_t, sum = threshold_loss(Lcard, threshold, output, total, sum)
                # loss_t.requires_grad_(True)
                loss.backward(retain_graph=True)
                # loss_t.backward()
                train_loss.append(loss.item())
                optimizer.step()

                pbar.set_description(f'Train Epoch: {epoch}/{EPOCH} loss: {np.mean(train_loss)}')
                ''' if batch_idx % 100 == 0:
                    correct = torch.zeros(1).squeeze().cuda()
                    total = torch.zeros(1).squeeze().cuda()
                    output_ = torch.sigmoid(model(data))
                    pred = torch.where(output_ >= 0.5, torch.ones_like(output_), torch.zeros_like(output_))
                    correct += (pred == label).sum()
                    total += label.shape[0]*label.shape[1]
                    print("epoch: ", epoch, "batch_idx: ", batch_idx, "accuracy: ",
                          correct/total*100) '''
                scheduler.step()
        test(criterion, epoch, optimizer)

'''    np.save('train_loss_cnn.npy', train_loss)
    np.save('test_loss_cnn.npy', test_loss)'''


s = time.time()
train()
print('consuming time: ', time.time() - s)