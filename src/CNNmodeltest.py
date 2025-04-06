import torch
import torch.nn as nn
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

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=10, padding='same')
        self.batch_norm1 = nn.BatchNorm1d(10)  # 添加批量归一化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)  # 添加丢弃层
        self.fc = nn.Linear((input_size // 2) * 10, num_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # 替换为LeakyReLU

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)  # 应用批量归一化
        x = self.leaky_relu(x)  # 使用LeakyReLU激活函数
        x = self.pool(x)
        x = self.dropout(x)  # 应用丢弃层
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
            labels = np.full((data.shape[0],), label_counter)  # 假设每个文件的标签相同
            
            # 对应的文件标注
            print(f"Class {label_counter}: {file_name}")

            data = data.astype(np.float32)
            all_data.append(data)
            all_labels.extend(labels)
            label_counter += 1  # 更新标签计数器
    
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
model = SimpleCNN(input_size=input_size, num_classes=num_classes)

# 加载模型权重
model.load_state_dict(torch.load("CNNbest_fault_diagnosis_model.pth"))
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
plt.title("Per-Class Accuracy of CNN Model")
plt.xticks(range(num_classes), [f"Class {i}" for i in range(num_classes)])
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
