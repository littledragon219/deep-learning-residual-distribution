import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)  # 设置随机种子为42

class DCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=10, padding=5)
        self.batch_norm1 = nn.BatchNorm1d(10)

        self.conv2 = nn.Conv1d(10, 10, kernel_size=10, padding=5)
        self.batch_norm2 = nn.BatchNorm1d(10)

        self.conv3 = nn.Conv1d(10, 10, kernel_size=10, padding=5)
        self.batch_norm3 = nn.BatchNorm1d(10)

        self.conv4 = nn.Conv1d(10, 10, kernel_size=10, padding=5)
        self.batch_norm4 = nn.BatchNorm1d(10)

        self.conv5 = nn.Conv1d(10, 10, kernel_size=10, padding=5)
        self.batch_norm5 = nn.BatchNorm1d(10)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # 提前定义全连接层
        dummy_input = torch.zeros(1, 1, input_size)
        dummy_output = self.pool(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))))
        self.fc_input_size = dummy_output.view(dummy_output.size(0), -1).size(1)
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.leaky_relu(x)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.leaky_relu(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


# 添加噪声的函数
def add_noise(data, snr_range):
    snr = np.random.uniform(*snr_range)
    signal_power = np.mean(data**2, axis=-1, keepdims=True)
    noise_power = signal_power / (10**(snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

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
            data = df.iloc[1:401, :500].values  # 样本数据
            if file_name not in label_mapping:
                label_mapping[file_name] = label_counter
                label_counter += 1
            labels = np.full((data.shape[0],), label_mapping[file_name])
            data = data.astype(np.float32)
            labels = labels.astype(np.int64)
            all_data.append(data)
            all_labels.extend(labels)

    return np.concatenate(all_data, axis=0), np.array(all_labels), len(label_mapping)

# 设置参数
input_size = 500
folder_path = 'D:/大三上学期/科研训练/'

# 加载数据
data, labels, num_classes = load_data_from_excel(folder_path)

# 应用 zero-padding 操作
def apply_zero_padding(data, target_length):
    return np.array([np.pad(row, (0, target_length - len(row)), 'constant') if len(row) < target_length else row[:target_length] for row in data])

padded_data = apply_zero_padding(data, input_size)

# 添加噪声
noisy_data = add_noise(padded_data, snr_range=(0, 8))

# 转换为 PyTorch 张量
inputs = torch.tensor(noisy_data, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
labels = torch.tensor(labels, dtype=torch.long)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(inputs, labels), batch_size=128, shuffle=True)  # mini-batch size=128

# 实例化模型
model = DCNN(input_size=input_size, num_classes=num_classes)
print(model)

# 设置训练参数
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 使用论文中的参数

# 训练循环并记录损失
def train_model(model, train_loader, epochs=100):  # epochs=100
    model.train()
    best_loss = float('inf')
    loss_history = []  # 记录损失
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        # 保存最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "DCNNnoisebest_model.pth")
    return loss_history

# 训练模型
loss_history = train_model(model, train_loader)

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='b', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid()
plt.show()