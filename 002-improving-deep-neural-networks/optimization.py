import numpy as np

# 가상의 데이터셋 생성
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])  # 입력 데이터
y = np.dot(X, np.array([1, 2])) + 3  # 실제 타겟 값

# 가중치(w)와 편향(b) 초기화
w = np.zeros(2)
b = 0

# 학습률과 에폭 설정
learning_rate = 0.01
epochs = 1000

# 경사 하강법을 사용한 최적화
for epoch in range(epochs):
    # 모델의 예측
    y_pred = np.dot(X, w) + b

    # 손실 계산 (MSE)
    loss = np.mean((y - y_pred) ** 2)

    # 기울기 계산
    dw = -(2 / len(X)) * np.dot(X.T, (y - y_pred))  # w에 대한 손실의 기울기
    db = -(2 / len(X)) * np.sum(y - y_pred)  # b에 대한 손실의 기울기

    # 파라미터 업데이트
    w -= learning_rate * dw
    b -= learning_rate * db

    # 일정 간격으로 진행 상황 출력
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# 최적화 후의 파라미터 출력
print(f'Optimized weights: {w}, Optimized bias: {b}')
