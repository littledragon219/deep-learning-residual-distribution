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
model = DCNN(input_size=input_size, num_classes=num_classes)

# 加载模型权重
model.load_state_dict(torch.load("DCNNnoisebest_model.pth"))  # 确保路径正确
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
plt.title("Per-Class Accuracy of DCNN Model")
plt.xticks(range(num_classes), [f"Class {i}" for i in range(num_classes)])
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()