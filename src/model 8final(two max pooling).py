import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import random
import matplotlib.pyplot as plt

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)  # 设置随机种子为42

# 定义残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# 定义完整模型
class FaultDiagnosisModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FaultDiagnosisModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=10, padding='same')
        self.bn1 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()
        self.res_block1 = ResidualBlock(10, 10, kernel_size=10)  # 第一个残差块
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 第一次最大池化
        self.res_block2 = ResidualBlock(10, 20, kernel_size=10, downsample=nn.Conv1d(10, 20, kernel_size=1))  # 第二个残差块
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 第二次最大池化
        self.fc = nn.Linear((input_size // 4) * 20, num_classes)  # 全连接层，输入大小调整为 input_size // 4

    def forward(self, x):
        print(f"Input: {x.size()}")  # 调试输出
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        print(f"After conv1: {x.size()}")  # 调试输出
        x = self.res_block1(x)
        print(f"After res_block1: {x.size()}")  # 调试输出
        x = self.pool1(x)
        print(f"After pool1: {x.size()}")  # 调试输出
        x = self.res_block2(x)
        print(f"After res_block2: {x.size()}")  # 调试输出
        x = self.pool2(x)  # 新增池化层
        print(f"After pool2: {x.size()}")  # 调试输出
        x = x.view(x.size(0), -1)  # 展平
        print(f"After flatten: {x.size()}")  # 调试输出
        x = self.fc(x)
        print(f"After fc: {x.size()}")  # 调试输出
        return nn.functional.softmax(x, dim=1)

# 加载数据函数
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

# 数据预处理
def apply_zero_padding(data, target_length):
    return np.array([np.pad(row, (0, target_length - len(row)), 'constant') if len(row) < target_length else row[:target_length] for row in data])

padded_data = apply_zero_padding(data, input_size)

# 转换为 PyTorch 张量
inputs = torch.tensor(padded_data, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
labels = torch.tensor(labels, dtype=torch.long)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(inputs, labels), batch_size=128, shuffle=True)

# 实例化模型
model = FaultDiagnosisModel(input_size=input_size, num_classes=num_classes)
print(model)

# 设置训练参数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
def train_model(model, train_loader, epochs=100):
    model.train()
    best_loss = float('inf')
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_fault_diagnosis_model.pth")
    return loss_history

# 开始训练
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
