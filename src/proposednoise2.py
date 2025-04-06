import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import pywt  # 小波变换库
from torch.cuda.amp import autocast
# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)  # 设置随机种子为42

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.SELU = nn.SELU()  # 使用SELU激活函数
        self.dropout = nn.Dropout(p=0.3)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.SELU(out1)
        out1 = self.dropout(out1)

        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = self.SELU(out2)
        out2 = self.dropout(out2)

        out3 = self.conv3(x)  # 使用第三个卷积层
        out = out1 + out2 + out3  # 跳跃连接
        out += identity
        out = self.SELU(out)
        out = self.dropout(out)

        return out

# 定义模型
class FaultDiagnosisModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FaultDiagnosisModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(10)
        self.SELU = nn.SELU()
        self.dropout = nn.Dropout(p=0.5)
        self.res_block1 = ResidualBlock(10, 10, kernel_size=10)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.res_block2 = ResidualBlock(10, 20, kernel_size=10, downsample=nn.Conv1d(10, 20, kernel_size=1))
        self.fc1 = nn.Linear((input_size // 2) * 20, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.SELU(x)
        x = self.dropout(x)
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.res_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 添加高斯噪声
def add_gaussian_noise(inputs, snr_db):
    signal_power = np.mean(inputs**2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), inputs.shape)
    return inputs + noise

# 自适应降噪：小波变换
def wavelet_denoising(data, wavelet='db1', level=1):
    denoised_data = []
    for row in data:
        coeffs = pywt.wavedec(row, wavelet, level=level)
        threshold = np.sqrt(2 * np.log(len(row)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        denoised_row = pywt.waverec(coeffs, wavelet)
        denoised_data.append(denoised_row[:len(row)])  # 截取原始长度
    return np.array(denoised_data)

# 加载数据
def load_data_from_excel(folder_path):
    all_data = []
    all_labels = []
    label_mapping = {}
    label_counter = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx') and 'train_data' in file_name:
            file_path = os.path.join(folder_path, file_name)
            print(f"Loading file: {file_path}")
            df = pd.read_excel(file_path, header=None)
            data = df.iloc[1:401, :500].values
            if file_name not in label_mapping:
                label_mapping[file_name] = label_counter
                label_counter += 1
            labels = np.full((data.shape[0],), label_mapping[file_name])
            data = data.astype(np.float32)
            labels = labels.astype(np.int64)
            all_data.append(data)
            all_labels.extend(labels)

    return np.concatenate(all_data, axis=0), np.array(all_labels), len(label_mapping)

# 数据预处理
input_size = 500
folder_path = 'D:/大三上学期/科研训练/'
data, labels, num_classes = load_data_from_excel(folder_path)

# 应用降噪
data = wavelet_denoising(data)

# 添加噪声
snr_range = [0, 2, 4, 6, 8]
noisy_data = np.vstack([add_gaussian_noise(data, snr_db) for snr_db in snr_range])
noisy_labels = np.tile(labels, len(snr_range))

# 转换为 PyTorch 张量
inputs = torch.tensor(noisy_data, dtype=torch.float32).unsqueeze(1)
labels = torch.tensor(noisy_labels, dtype=torch.long)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(inputs, labels), batch_size=128, shuffle=True)

# 实例化模型
model = FaultDiagnosisModel(input_size=input_size, num_classes=num_classes)
print(model)

# 定义优化器和学习率调度器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        bce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-bce_loss)  # 计算预测概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss

criterion = FocalLoss()

def train_model_with_cpu_amp(model, train_loader, epochs=15, device='cpu'):
    model = model.to(device)  # 将模型移动到CPU
    best_loss = float('inf')
    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()

            # 使用 autocast 来开启 AMP
            with autocast():
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()  # 调整学习率
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # 保存最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "proposednoisebest_fault_diagnosis_model_cpu_amp.pth")

    return loss_history


device = 'cpu'
loss_history = train_model_with_cpu_amp(model, train_loader, epochs=20, device=device)


# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='b', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid()
plt.show()
