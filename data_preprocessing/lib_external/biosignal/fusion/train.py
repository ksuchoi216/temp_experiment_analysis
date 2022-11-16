import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

import models


def set_seed(seed):
    # Random Seed 설정
    # 실험 환경을 동일하게 맞추기 위해서 Random Seed를 고정합니다.
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
            data['s'] = np.load(file)
        self.data = data

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        x = self.data['x'][idx]
        f = self.data['f'][idx]
        s = self.data['s'][idx]
        return x, f, s


def train_signal_reconstruction(
    data_path='./data',
    logs_path='./logs',
    seed=1234,
    epochs=10,
    batch_size=128,
    learning_rate=0.001,
    weight_decay=0.00001,
    gpu_index=0,
):
    """
    Signal Reconstruction Model Training

    data_path : 학습에 사용할 데이터 경로
    logs_path : 학습 결과 저장 경로
    gpu_index : 학습에 사용할 GPU index
    """

    set_seed(seed)
    os.makedirs(logs_path, exist_ok=True)

    # 학습과 검증에 사용할 데이터셋을 준비합니다.
    train_dataset = Dataset(os.path.join(data_path, 'train.npy'))
    val_dataset = Dataset(os.path.join(data_path, 'val.npy'))

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

    device = torch.device('cuda:%d' %
                          gpu_index if torch.cuda.is_available() else 'cpu')
    net = models.get_signal_reconstruction_model()
    net = net.to(device)
    # print(net)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='mean')

    print('%12s %12s %12s %12s' % ('Epoch', 'Time', 'Train Loss', 'Val Loss'))

    time_total = 0
    for epoch in range(epochs):

        # Train
        t0 = time.time()
        net = net.train()
        losses = 0
        for data in train_loader:
            x = data[0].to(device)
            s = data[2].to(device)

            optimizer.zero_grad()

            out = net(x)
            out = out.squeeze()
            loss = criterion(out, s)

            loss.backward()
            optimizer.step()

            losses += loss.detach()
        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        # Evaluate
        t0 = time.time()
        net = net.eval()
        losses = 0
        for data in val_loader:
            x = data[0].to(device)
            s = data[2].to(device)

            with torch.no_grad():
                out = net(x)
                out = out.squeeze()
                loss = criterion(out, s)

                losses += loss.detach()
        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total += (time_train + time_val)

        print('%12d %12.4f %12.4f %12.4f' %
              (epoch+1, time_total, loss_train, loss_val))

    # Save the model
    model_file = os.path.join(logs_path, 'model_sr.pth')
    torch.save(net.cpu().state_dict(), model_file)
    print('Model -> ', model_file)


def train_frequency_estimation(
    data_path='./data',
    logs_path='./logs',
    seed=1234,
    epochs=10,
    batch_size=128,
    learning_rate=0.001,
    weight_decay=0.00001,
    gpu_index=0,
):
    """
    Frequency Estimation Model Training

    data_path : 학습에 사용할 데이터 경로
    logs_path : 학습 결과 저장 경로
    gpu_index : 학습에 사용할 GPU index
    """

    set_seed(seed)
    os.makedirs(logs_path, exist_ok=True)

    # 학습과 검증에 사용할 데이터셋을 준비합니다.
    train_dataset = Dataset(os.path.join(data_path, 'train.npy'))
    val_dataset = Dataset(os.path.join(data_path, 'val.npy'))

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

    device = torch.device('cuda:%d' %
                          gpu_index if torch.cuda.is_available() else 'cpu')
    #net = models.get_signal_reconstruction_model()
    net = models.get_frequency_estimation_model()
    net = net.to(device)
    # print(net)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #criterion = nn.MSELoss(reduction='mean')
    criterion = nn.L1Loss(reduction='mean')

    print('%12s %12s %12s %12s' % ('Epoch', 'Time', 'Train Loss', 'Val Loss'))

    time_total = 0
    for epoch in range(epochs):

        # Train
        t0 = time.time()
        net = net.train()
        losses = 0
        for data in train_loader:
            x = data[0].to(device)
            f = data[1].to(device)

            optimizer.zero_grad()

            out = net(x)
            loss = criterion(out.squeeze(), f.unsqueeze(1))

            loss.backward()
            optimizer.step()

            losses += loss.detach()
        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        # Evaluate
        t0 = time.time()
        net = net.eval()
        losses = 0
        for data in val_loader:
            x = data[0].to(device)
            f = data[1].to(device)

            with torch.no_grad():
                out = net(x)
                loss = criterion(out.squeeze(), f.unsqueeze(1))

                losses += loss.detach()
        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total += (time_train + time_val)

        print('%12d %12.4f %12.4f %12.4f' %
              (epoch+1, time_total, loss_train, loss_val))

    # Save the model
    model_file = os.path.join(logs_path, 'model_fe.pth')
    torch.save(net.cpu().state_dict(), model_file)
    print('Model -> ', model_file)


def train():
    train_signal_reconstruction()
    train_frequency_estimation()


if __name__ == '__main__':
    train()
