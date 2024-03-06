import pandas as pd
import numpy as np

# 读取Excel文件
data = pd.read_excel('KGE.xlsx', engine='openpyxl')

# 获取观测值和模型预测值列
observed = data['Observations']
predicted = data['Model_Predictions']

# 计算均值和标准差
mean_observed = observed.mean()
std_observed = observed.std()
mean_predicted = predicted.mean()
std_predicted = predicted.std()

# 计算KGE
kge = 1 - np.sqrt((0.25 * (mean_observed - mean_predicted)**2 +
                  0.25 * (std_observed / std_predicted - 1)**2 +
                  0.25 * (np.corrcoef(observed, predicted)[0, 1] - 1)**2))

print("KGE:", kge)
