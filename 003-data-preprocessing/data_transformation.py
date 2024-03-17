# 라이브러리 없이 파이썬 기본 기능만 사용하여 구현
import numpy as np
import pandas as pd

# 1. Min-Max 정규화
def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# 2. Z-점수 표준화 (Standardization)
def standardize(data):
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    return [(x - mean) / std_dev for x in data]

# 3. 로그 변환
def log_transform(data):
    return [np.log(x) for x in data]

# 4. 제곱근 변환
def sqrt_transform(data):
    return [np.sqrt(x) for x in data]

# 5. 박스-콕스 변환 (람다 = 0.5로 가정)
def box_cox_transform(data, lmbda=0.5):
    return [(x ** lmbda - 1) / lmbda if lmbda != 0 else np.log(x) for x in data]

# 데이터 변환 실행
data = [10, 20, 30, 40, 50]
data_normalized = min_max_normalize(data)
data_standardized = standardize(data)
data_logged = log_transform(data)
data_sqrt = sqrt_transform(data)
data_boxcox = box_cox_transform(data)

# 변환 결과 생성
results_python = pd.DataFrame({
    "Original": data,
    "Normalized": data_normalized,
    "Standardized": data_standardized,
    "Logged": data_logged,
    "Square Root": data_sqrt,
    "Box-Cox": data_boxcox
})

print(results_python)

# 라이브러리를 사용하여 데이터 변환 구현
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 예제 데이터 생성
data_example = np.array([[10], [20], [30], [40], [50]])

# 1. Min-Max 정규화
min_max_scaler = MinMaxScaler()
data_normalized = min_max_scaler.fit_transform(data_example)

# 2. Z-점수 표준화 (Standardization)
standard_scaler = StandardScaler()
data_standardized = standard_scaler.fit_transform(data_example)

# 3. 로그 변환
data_logged = np.log(data_example)

# 4. 제곱근 변환
data_sqrt = np.sqrt(data_example)

# 5. 박스-콕스 변환
data_boxcox = np.array([stats.boxcox(data_example.flatten())[0]]).reshape(-1, 1)

# 결과 출력을 위해 DataFrame으로 변환
results = pd.DataFrame({
    "Original": data_example.flatten(),
    "Normalized": data_normalized.flatten(),
    "Standardized": data_standardized.flatten(),
    "Logged": data_logged.flatten(),
    "Square Root": data_sqrt.flatten(),
    "Box-Cox": data_boxcox.flatten()
})

print(results)

# 라이브러리 없이 파이썬 기본 기능만 사용하여 원-핫 인코딩 구현
def one_hot_encoding(data):
    categories = sorted(set(data)) # ['blue', 'green', 'red']
    encoded = []
    for d in data:
        encoded.append([int(d == category) for category in categories])
    return encoded

data_example = ['red', 'green', 'blue', 'red']

encoded_data = one_hot_encoding(data_example)
print(encoded_data)

# 라이브러리를 사용하여 원-핫 인코딩 구현
df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red']})
encoded_df = pd.get_dummies(df, columns=['color'])
print(encoded_df.astype(int))

# 텐서 변환
import torch

tensor = torch.tensor(encoded_data, dtype=torch.float32)
print(tensor)

# 배치 로딩
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 10)
        self.labels = torch.randint(0, 2, (100,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset()

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_idx, (data, labels) in enumerate(dataloader):
    # 학습 코드
    print(f"Batch {batch_idx + 1}, Data shape: {data.shape}, Labels shape: {labels.shape}, Data: {data}, Labels: {labels}")