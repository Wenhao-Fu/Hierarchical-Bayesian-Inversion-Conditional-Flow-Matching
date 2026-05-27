import os
import torch
from torch.utils.data import DataLoader, random_split
from load_data import load_data


def l2_loss(pred, true):
    loss = torch.sum((pred-true)**2, dim=[1, 2, 3])
    return torch.mean(loss)


def get_train_data(args):
    perm, label = load_data(args.training_data_path)
    perm = torch.as_tensor(perm.reshape(args.train_number, 1, 64, 64))
    label = torch.as_tensor(label)
    dataset = torch.utils.data.TensorDataset(perm, label)

    # 计算训练集和验证集的大小
    train_size = int(args.training_rate * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(42)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False)

    return train_loader, val_loader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
