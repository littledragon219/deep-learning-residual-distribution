import pandas as pd
import numpy as np
import os
import scipy.io
import re

# 定义文件夹路径
folder_path = 'D:/大三上学期/科研训练/'

# 获取所有以 normal 开头的 MAT 文件
file_list = [f for f in os.listdir(folder_path) if f.startswith('outerrace021') and f.endswith('.mat')]

# 设定采样频率
sampling_rate = 12000  # 12kHz
window_size = 500  # 窗口大小
overlap = 250  # 窗口重叠大小

# 创建空的 DataFrame 来存储所有训练集和测试集
all_train_data = pd.DataFrame()
all_test_data = pd.DataFrame()

# 遍历每个文件
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    
    # 读取 MAT 文件
    mat_data = scipy.io.loadmat(file_path)
    
    # 获取所有字段名
    keys = mat_data.keys()
    
    # 过滤掉不需要的字段
    data_keys = [key for key in keys if not key.startswith('__')]
    
    # 提取符合规律的字段名
    data_columns = [key for key in data_keys if re.match(r'X\d{3}(_DE_time)?', key)]
    
    # 假设我们只取第一个符合条件的字段作为数据
    if data_columns:
        data = mat_data[data_columns[0]].flatten()  # 使用第一个字段，并展平为一维数组
        print(f"Processing file: {file_name}, Total data points: {len(data)}")
        
        # 计算可生成的样本数量
        num_samples = len(data) // window_size
        print(f"Number of samples that can be generated from {file_name}: {num_samples}")
        
        if num_samples >= 200:  # 确保样本数量足够
            # 分割数据成样本
            samples = np.array([data[i:i + window_size] for i in range(0, len(data) - window_size + 1, window_size-overlap)])
            
            # 随机抽取训练集和测试集索引
            np.random.seed(42)
            train_ratio=0.5
            num_train_samples = int(num_samples * train_ratio)
            indices = np.random.permutation(num_samples)
            train_indices = np.random.choice(num_samples, num_train_samples, replace=False)
            test_indices = np.setdiff1d(np.arange(num_samples), train_indices)
            
            # 提取训练集和测试集
            train_samples = samples[train_indices]
            test_samples = samples[test_indices]
            
            # 构造 DataFrame
            train_df = pd.DataFrame(train_samples, columns=[f'Point_{i+1}' for i in range(window_size)])
            train_df['Type'] = 'Train'
            
            test_df = pd.DataFrame(test_samples, columns=[f'Point_{i+1}' for i in range(window_size)])
            test_df['Type'] = 'Test'
            
            # 添加文件来源信息
            train_df['Source_File'] = file_name
            test_df['Source_File'] = file_name
            
            # 将数据追加到总 DataFrame
            all_train_data = pd.concat([all_train_data, train_df], ignore_index=True)
            all_test_data = pd.concat([all_test_data, test_df], ignore_index=True)
        else:
            print(f"Not enough samples to extract from {file_name}.")
    else:
        print(f"No valid data columns found in {file_name}.")

# 将训练集和测试集分别写入 Excel 文件
train_output_file = 'outerrace021train_data.xlsx'
test_output_file = 'outerrace021test_data.xlsx'

all_train_data.to_excel(train_output_file, index=False)
all_test_data.to_excel(test_output_file, index=False)

print(f"Training data has been written to {train_output_file}.")
print(f"Testing data has been written to {test_output_file}.")