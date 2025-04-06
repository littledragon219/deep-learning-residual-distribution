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

class DNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.bn1 = nn.BatchNorm1d(1000)  # 添加批归一化
        self.fc2 = nn.Linear(1000, 500)
        self.bn2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.output_layer = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout统一放置
        self.activation = nn.LeakyReLU()  # 替换ReLU为LeakyReLU

        # 权重初始化
        for layer in [self.fc1, self.fc2, self.fc3, self.output_layer]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.output_layer(x)
        return nn.functional.softmax(x, dim=1)


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

# 应用 zero-padding 操作
def apply_zero_padding(data, target_length):
    return np.array([np.pad(row, (0, target_length - len(row)), 'constant') if len(row) < target_length else row[:target_length] for row in data])

# 设置参数
input_size = 500
folder_path = 'D:/大三上学期/科研训练/'

# 加载数据
data, labels, num_classes = load_data_from_excel(folder_path)

# 应用 zero-padding
padded_data = apply_zero_padding(data, input_size)

# 转换为 PyTorch 张量
inputs = torch.tensor(padded_data, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
labels = torch.tensor(labels, dtype=torch.long)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(inputs, labels), batch_size=128, shuffle=True)

# 实例化模型
model = DNNModel(input_size=input_size, num_classes=num_classes)
print(model)

# 设置训练参数
criterion = nn.CrossEntropyLoss()  # 损失函数
# 调整学习率：添加学习率调度器
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 训练循环
def train_model(model, train_loader, epochs=1000):
    model.train()
    best_loss = float('inf')
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)  # Flatten inputs
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        scheduler.step(epoch_loss)  # 调整学习率

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "improved_DNN_model.pth")
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
