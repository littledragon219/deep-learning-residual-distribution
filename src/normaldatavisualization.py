import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import re

# 定义文件夹路径
folder_path = 'D:/大三上学期/科研训练/normal'

# 获取所有MAT文件
file_list = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

# 设定采样频率
sampling_rate = 12000  # 12kHz

# 创建一个空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 遍历每个文件
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    
    # 读取MAT文件
    mat_data = scipy.io.loadmat(file_path)
    
    # 获取所有字段名
    keys = mat_data.keys()
    
    # 过滤掉不需要的字段（如 '__header__', '__version__', '__globals__'）
    data_keys = [key for key in keys if not key.startswith('__')]
    
    # 提取符合规律的字段名
    data_columns = [key for key in data_keys if re.match(r'X\d{3}(_DE_time)?', key)]
    
    # 假设我们只取第一个符合条件的字段作为数据
    if data_columns:
        data = mat_data[data_columns[0]]  # 使用第一个符合条件的字段名
        data = pd.DataFrame(data)  # 将数据转换为DataFrame
        
        # 查看数据的前几行
        print(f"Processing file: {file_name}")
        print(data.head())
        
        # 只选择前100个数据点
        data_subset = data.head(100)
        
        # 生成时间轴
        time = np.arange(len(data_subset)) / sampling_rate
        
        # 绘制所有列的数据
        plt.figure(figsize=(12, 6))
        for column in data_subset.columns:
            plt.plot(time, data_subset[column], label=column)

        plt.title(f'Vibration Signals Visualization - {file_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
        # 显示图表
        plt.show()

        # 数据分析示例
        for column in data_subset.columns:
            mean_value = data_subset[column].mean()
            median_value = data_subset[column].median()
            std_dev = data_subset[column].std()
            
            print(f'Analysis for {column} in {file_name}:')
            print(f'Mean: {mean_value:.2f}')
            print(f'Median: {median_value:.2f}')
            print(f'Standard Deviation: {std_dev:.2f}')
            print('---')

        # 将数据添加到总的DataFrame中
        data_subset['Source File'] = file_name  # 添加一列以标识数据来源
        all_data = pd.concat([all_data, data_subset], ignore_index=True)
    else:
        print(f"No valid data columns found in {file_name}.")

# 将所有数据写入Excel文件
output_file = 'normal.xlsx'
all_data.to_excel(output_file, index=False)

print(f"All data has been written to {output_file}.")