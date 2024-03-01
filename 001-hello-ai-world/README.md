# hello AI world
농업 혁명, 산업 혁명, 인터넷 혁명에 이어 AI 혁명이 왔다. 소용돌이 같은 변화 속에서 흐름을 정신없이 좇다보면 흐릿한 사고 속에서 혼란만 가중된다. 이럴 때일수록 의심할 여지가 없는 기본 개념을 확실히 다져놓는게 결국 빠른 길일거라는 믿음 아래 기록을 시작해본다!

인공지능은 말그대로 인간의 지능을 기계로 모방해 구현해 놓은 걸 말한다.

> 왜 인간의 지능을 모방하려고 할까?

컴퓨터의 주된 역할은 문제 해결이다. 하지만, 물리법칙을 찾아내고 새로운 기술을 개발하는건 인간이 해왔다. 이러한 복잡하고 추상적인 개념을 이해하고 구체화시키는 일을 할 수 있는 인간의 뇌의 작동방식을 컴퓨터에 적용시킨다면 컴퓨터도 인간과 같은 일을 할 수 있지 않을까 하는 생각에서 출발한다.

*인공지능은 합리적인 인간을 모델로 삼는다.*
 
머신러닝은 인공지능의 한 분야로, 경험(데이터)을 통해 스스로 개선하는 방식이다. 머신러닝은 크게 지도학습, 비지도학습, 강화학습으로 나뉜다. 여기선 간단한 개념정리만 해둔다.

- **지도학습(Supervised Learning)**: 답이 주어진 여러 예제를 통해 학습하는 방식
- **비지도학습(Unsupervised Learning)**: 답이 주어지지 않은 여러 예제에서 패턴을 찾아 학습하는 방식
- **강화학습(Reinforcement Learning)**: 시행착오와 그에 따른 보상과 처벌 체계를 통해 학습하는 방식

> 그래서 어떻게 쓰는건데?

바로 본론으로 들어가서 머신러닝의 전체적인 워크플로우를 살펴보자.  

1. 어떤 문제를 해결할건지 명확히 해야한다. 주가 예측을 예시로 들어보자. 과거의 주가 데이터는 이미 제공된 값이다. 따라서 답이 주어진 여러 예시를 통해 학습하는 지도학습 방식을 선택할 수 있다.
2. 데이터를 수집하고, 전처리한다.
3. 모델을 설계하고, 학습시킨다.
4. 모델을 평가하고, 사용해 예측을 한다.
5. 예측 결과를 분석하고, 모델을 개선한다.

> 어떤 언어를 사용해야하나?

파이썬이 대세다. 파이썬은 데이터 분석과 머신러닝에 대한 라이브러리가 잘 갖춰져 있고, 사용하기 쉽다. 라이브러리로는 PyTorch를 사용한다. PyTorch는 페이스북에서 만든 딥러닝 라이브러리로, 유연하고 파이썬 기반으로 작동한다.

#### 인공신경망 모델
데이터 수집과 전처리 과정은 나중에 자세히 알아보도록 하고, 인공신경망 모델의 작동 흐름을 좇아가보자.

뇌는 여러 층의 수많은 뉴런들이 시냅스로 얽혀 정보를 처리하고, 인공신경망은 이런 뇌의 구조를 모방한다. 인공신경망에서 층은 레이어, 뉴런은 노드, 시냅스는 노드 간 연결을 말한다. 뇌의 정보의 처리 과정은 순전파에 해당하고, 학습 과정은 역전파에 해당한다.

각 노드엔 `weight`(가중치)와 `bias`(편향)가 있다. 이를 `parameters`(매개변수)라고 하고 훈련이 진행됨에 따라 계속해서 수정되는 값이다. 기본적인 선형 회귀 모델은 *y = mx + b*이고, 여기서 m이 가중치, b가 편향이다. 이 m과 b를 반복 수정해 손실을 최소화시킨다.  데이터가 입력으로 들어와 노드들이 있는 여러 층을 거쳐 예측값이 출력이 되는데 이 방향의 흐름을 `forward propagation`(순전파)이라고 한다. 이렇게 나온 예측치와 레이블(정답)의 오차를 계산하는 과정이 `loss function`이고, 이 오차를 최소화하기 위해 역방향으로 각 노드에 있는 weights와 biases를 수정해나가는걸 `optimizing`이라고 한다. 또한, 이런 역방향의 흐름을 `backward propagation`이라고 한다.

이 전체 흐름을 코드로 살펴보자.

```commandline
# Create a Model
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return x
        
# Instantiate the model
simpy = SimpleNeuralNetwork(1, 1)

# Create a simple dataset
X = torch.tensor([[1.0], [2.0]])
y_hat = torch.tensor([[2.0], [4.0]])

# Print the model parameters
print(f"Initial weights: {simpy.linear.weight}")
print(f"Initial bias: {simpy.linear.bias}")

# Make a prediction
y = simpy(X)
print(f"Prediction before training: {y}")

# Calculate the loss
criterion = nn.MSELoss()
loss = criterion(y, y_hat)
print(f"Loss: {loss}")

# Optimize the model
optimizer = torch.optim.SGD(simpy.parameters(), lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Print the model parameters
print(f"Updated weights: {simpy.linear.weight}")
print(f"Updated bias: {simpy.linear.bias}")

# Make a prediction
y = simpy(X)
print(f"Prediction after training: {y}")
print(f"Loss: {criterion(y, y_hat)}")        
```