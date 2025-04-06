import torch
import torch.nn as nn
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

# 定义残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True) 
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample  # 线性投影

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)  # 用于调整维度
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  
        out = self.dropout(out)  # 应用Dropout
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 跳跃连接
        out = self.relu(out)  
        out = self.dropout(out)  # 应用Dropout
        return out


class FaultDiagnosisModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FaultDiagnosisModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=10, padding='same')  # 初始卷积
        self.bn1 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()  
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层
        self.res_block1 = ResidualBlock(10, 10, kernel_size=10)  # 通道保持一致
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 最大池化
        self.res_block2 = ResidualBlock(10, 20, kernel_size=10, downsample=nn.Conv1d(10, 20, kernel_size=1))  # 改变通道数
        self.fc = nn.Linear((input_size // 2) * 20, num_classes)  # 调整输入大小

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.dropout(x)  # 应用Dropout
        x = self.res_block1(x)
        x = self.relu(x)  
        x = self.dropout(x)  # 应用Dropout
        x = self.pool1(x)
        x = self.res_block2(x)
        x = self.relu(x)  
        x = self.dropout(x)  # 应用Dropout
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        x = self.relu(x) 
        x = nn.functional.softmax(x, dim=1)
        return x

# 加载测试数据
def load_test_data_from_excel(folder_path):
    all_data = []
    all_labels = []
    label_mapping = {}
    label_counter = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx') and 'test_data' in file_name:
            file_path = os.path.join(folder_path, file_name)
            print(f"Loading file: {file_path}")
            df = pd.read_excel(file_path, header=None)
            data = df.iloc[1:401, :500].values  # 样本数据
            if file_name not in label_mapping:
                label_mapping[file_name] = label_counter
                label_counter += 1
            labels = np.full((data.shape[0],), label_mapping[file_name])
            data = data.astype(np.float32)
            all_data.append(data)
            all_labels.extend(labels)

    return np.concatenate(all_data, axis=0), np.array(all_labels)

# 设置参数
input_size = 500
folder_path = 'D:/大三上学期/科研训练/'

# 加载测试数据
test_data, test_labels = load_test_data_from_excel(folder_path)

# 应用 zero-padding 操作
def apply_zero_padding(data, target_length):
    return np.array([np.pad(row, (0, target_length - len(row)), 'constant') if len(row) < target_length else row[:target_length] for row in data])

padded_test_data = apply_zero_padding(test_data, input_size)

# 转换为 PyTorch 张量
test_inputs = torch.tensor(padded_test_data, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 创建数据加载器
test_loader = DataLoader(TensorDataset(test_inputs, test_labels), batch_size=128, shuffle=False)

# 实例化模型并加载权重
num_classes = 10  # 根据您的训练数据设置
model = FaultDiagnosisModel(input_size=input_size, num_classes=num_classes)

# 加载模型权重
model.load_state_dict(torch.load("proposednoisebest_fault_diagnosis_model.pth"))
model.eval()  # 设置为评估模式

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 计算每个类别的准确率和整体测试准确率
class_correct = np.zeros(num_classes)
class_total = np.zeros(num_classes)
total_loss = 0.0
total_samples = 0

with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        total_loss += loss.item() * batch_inputs.size(0)  # 累加损失
        total_samples += batch_inputs.size(0)
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别
        for label, prediction in zip(batch_labels, predicted):
            if label == prediction:
                class_correct[label] += 1
            class_total[label] += 1

# 计算每个类别的准确率
class_accuracies = class_correct / class_total * 100
overall_accuracy = sum(class_correct) / sum(class_total) * 100
average_loss = total_loss / total_samples

print(f"Overall Accuracy: {overall_accuracy:.2f}%")
print(f"Average Loss: {average_loss:.4f}")
for i, acc in enumerate(class_accuracies):
    print(f"Class {i} Accuracy: {acc:.2f}%")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.bar(range(num_classes), class_accuracies, color='skyblue')
plt.xlabel("Class Label")
plt.ylabel("Accuracy (%)")
plt.title("Per-Class Accuracy of Fault Diagnosis Model")
plt.xticks(range(num_classes), [f"Class {i}" for i in range(num_classes)])
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()