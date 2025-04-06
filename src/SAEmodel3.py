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

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 批量归一化
            nn.LeakyReLU()  # 使用LeakyReLU激活函数
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 定义堆叠自编码器（SAE）
class StackedAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(StackedAutoEncoder, self).__init__()
        self.ae1 = AutoEncoder(input_size, hidden_sizes[0])
        self.ae2 = AutoEncoder(hidden_sizes[0], hidden_sizes[1])
        self.ae3 = AutoEncoder(hidden_sizes[1], hidden_sizes[2])
        self.dropout = nn.Dropout(p=0.3)  # 增加Dropout几率为30%
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[2], num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        encoded1, _ = self.ae1(x)
        encoded1 = self.dropout(encoded1)
        encoded2, _ = self.ae2(encoded1)
        encoded2 = self.dropout(encoded2)
        encoded3, _ = self.ae3(encoded2)
        logits = self.classifier(encoded3)
        return logits

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
hidden_sizes = [1000, 500, 100]  # 增加隐藏层的神经元数量
folder_path = 'D:/大三上学期/科研训练/'

# 加载数据
data, labels, num_classes = load_data_from_excel(folder_path)

# 应用 zero-padding 操作
def apply_zero_padding(data, target_length):
    return np.array([np.pad(row, (0, target_length - len(row)), 'constant') if len(row) < target_length else row[:target_length] for row in data])

padded_data = apply_zero_padding(data, input_size)

# 打乱数据
indices = np.arange(padded_data.shape[0])
np.random.shuffle(indices)
shuffled_data = padded_data[indices]
shuffled_labels = labels[indices]

# 转换为 PyTorch 张量
inputs = torch.tensor(shuffled_data, dtype=torch.float32)
labels = torch.tensor(shuffled_labels, dtype=torch.long)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(inputs, labels), batch_size=128, shuffle=True)

# 实例化模型
model = StackedAutoEncoder(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)
print(model)

# 设置训练参数
criterion = nn.CrossEntropyLoss()  # 损失函数
# 使用AdamW优化器，并添加学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加权重衰减
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每10个epoch学习率减半

# 训练循环并记录损失
def train_model(model, train_loader, epochs=500):  # epochs=500
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
        scheduler.step()  # 更新学习率
        # 保存最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_stacked_autoencoder_model.pth")
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