import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

import models


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = {}
        with open(data_path, 'rb') as file:
            data['x'] = np.load(file)
            data['f'] = np.load(file)
            data['y'] = np.load(file)
        self.data = data

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        x = self.data['x'][idx]
        p = self.data['y'][idx]
        return x, p


def train(
    data_path='./data',
    logs_path='./logs',
    seed=1234,
    epochs=10,
    batch_size=128,
    learning_rate=0.0003,
):
    set_seed(seed)
    os.makedirs(logs_path, exist_ok=True)

    # 학습과 검증에 사용할 데이터셋을 준비합니다.
    train_dataset = Dataset(os.path.join(data_path, 'train.npy'))
    val_dataset = Dataset(os.path.join(data_path, 'val.npy'))
    test_dataset = Dataset(os.path.join(data_path, 'test.npy'))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = models.Classifier()
    net = net.to(device)
    print(net)

    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    print('%12s %12s %12s %12s %12s %12s' %
          ('Epoch', 'Time', 'Train Loss', 'Val Loss', 'Train Acc.', 'Val Acc.'))

    time_total = 0
    for epoch in range(epochs):
        # Train
        t0 = time.time()
        net = net.train()
        losses = 0
        corrects = 0
        for x, y in train_loader:
            x = x.to(device).unsqueeze(1)  # add channel dimension
            y = y.to(device)

            optimizer.zero_grad()

            out = net(x)
            out = out.mean(2).squeeze()  # global averaging
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            losses += loss.detach()
            correct = torch.sum((out > 0.5) == (y > 0.5))
            corrects += correct.detach()
        loss_train = losses / len(train_loader)
        acc_train = corrects / len(train_loader.dataset)
        t1 = time.time()
        time_train = t1 - t0

        # Evaluate
        t0 = time.time()
        net = net.eval()
        losses = 0
        corrects = 0
        for x, y in val_loader:
            x = x.to(device).unsqueeze(1)  # add channel dimension
            y = y.to(device)

            with torch.no_grad():
                out = net(x)
                out = out.mean(2).squeeze()  # global averaging
                loss = criterion(out, y)

                losses += loss.detach()
                correct = torch.sum((out > 0.5) == (y > 0.5))
                corrects += correct.detach()
        loss_val = losses / len(val_loader)
        acc_val = corrects / len(val_loader.dataset)

        t1 = time.time()
        time_val = t1 - t0

        time_total += (time_train + time_val)

        print('%12d %12.4f %12.4f %12.4f %12.4f %12.4f' %
              (epoch+1, time_total, loss_train, loss_val, acc_train, acc_val))

    # Test
    net = net.eval()
    losses = 0
    corrects = 0
    for x, y in test_loader:
        x = x.to(device).unsqueeze(1)  # add channel dimension
        y = y.to(device)

        with torch.no_grad():
            out = net(x)
            out = out.mean(2).squeeze()  # global averaging
            loss = criterion(out, y)

            losses += loss.detach()
            correct = torch.sum((out > 0.5) == (y > 0.5))
            corrects += correct.detach()
    loss_test = losses / len(test_loader)
    acc_test = corrects / len(test_loader.dataset)
    print('Test Loss -> %8.4f  |  Accuracy -> %8.4f' % (loss_test, acc_test))

    # Save the model
    model_file = os.path.join(logs_path, 'model.pth')
    torch.save(net.cpu().state_dict(), model_file)
    print('Model -> ', model_file)


if __name__ == '__main__':
    train()
