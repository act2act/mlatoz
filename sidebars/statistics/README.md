## Statistics
[효과적인 학습에 대한 짧은 글](../learning/README.md#about-learning)

### 통계학이란?
![statistic](./images/statistic.png)
불확실한 사실이나 대상에 대한 의사결정을 위한 결론이나 일반성을 이끌어내기 위해 관련된 데이터를 수집, 요약, 해석하는 데 필요한 이론이나 방법을 과학적으로 제시하는 학문이다.

- **기술 통계학(Descriptive Statistics)**: 데이터를 수집하고 정리하여 그림이나 표로 요약하거나 대표값이나 분포, 변동의 크기 등을 구하는 방법을 다룬다.
- **추론 통계학(Inferential Statistics)**: 통계학으로 얻은 결과가 미래에도 지속될 것인지를 예측하고 이들의 믿을 만한 정도를 평가하기 위해 가설을 세우고 이를 검정하는 방법을 다룬다.

### 자료의 수집
사회 현상을 조사하는 사회과학 분야에선 표집조사(sampling)와 여론조사(survey)를 통해 데이터를 수집한다.
모의 실험이나 측정을 통해 정보를 얻는 자연과학 분야에선 실험계획(experimental design) 방법을 사용한다.

사회과학, 자연과학 두 분야에서 자료를 수집하는 방법은 관심 있는 대상 전체를 조사하는 **전수조사**와 일부만 조사하는 **표본조사**가 있다.
예를 들어, 전국민을 대상으로 한 시청률 조사가 있다. 이는 전수조사이며 여기서 전국민은 통계학에서 **모집단**(population), 시청률은 **모수**(parameter)라고 한다.
반면, 전국민 중 일부만을 대상으로 한 시청률 조사라면, 이는 표본조사이며 여기서 조사 대상을 **표본**(sample), 시청률을 **통계량**(statistics)라고 한다.

통계 자료 분석의 전체적인 흐름은 다음과 같다.

> 모집단 -> 표본 -> 요약 및 정리(기술통계학) -> 분석 및 추론(추론통계학) -> 의사결정

### 자료의 이해
자료의 구성요소는 **변수**(variable)와 **개체**(observation)이다.
변수는 수집하는 자료에서 관심이 되고, 측정해야 할 특성을 의미하고, 개체는 실제 측정되는 대상으로 가장 기본이 되는 단위를 개체라고 한다.

자료는 변수의 성질에 따라 **질적 자료**(qualitative data)와 **양적 자료**(quantitative data)로 나뉜다.
질적 자료는 개체인 측정대상이 어느 범주에 들어가는지를 나타내고, **범주형 자료**(categorical data)라고도 한다.
양적 자료는 변수를 수치로 나타내는 자료로, 셀 수 있다면 **이산형 자료**(discrete data), 셀 수 없다면 **연속형 자료**(continuous data)로 나눈다.
질적 자료도 순서가 중요하다면 **순서형 자료**(ordinal data), 순서는 관계 없이 구분만을 위한다면 **명목형 자료**(nominal data)로 나눌 수 있다.

### 범주형 자료 요약
범주형 자료는 **도수분포표**(frequency table)를 통해 요약할 수 있다.
도수분포표는 자료를 범주별로 나누어 각 범주에 속하는 개체의 수를 나타낸 표이다.

이산형 자료는 셀 수 있는 경우엔 범주형 자료와 같이 도수분포표를 사용한 요약이 가능하지만, 셀 수 없는 경우엔 범주가 무한히 많아질 수 있기 때문에 연속형 자료 요약 방법을 사용한다.

### 연속형 자료 요약
연속형 자료는 보통 관측값의 최소값부터 최대값을 포함한 범위를 일정한 구간으로 나누어 각 구간에 포함되는 관측값의 개수를 도수로 나타낸 도수분포표를 사용한다.
이때, 나누어진 각 구간을 계급(class)이라고 하며, 각 계급을 포함한 범위를 **계급구간**(class interval)이라고 한다.

이를 그림이나 표를 이용한 시각화를 통해 요약하는 방법을 **히스토그램**(histogram)이라고 한다.

하지만, 도수분포표나 히스토그램만으로는 통계적 추론을 할만큼의 일관성이나 객관성을 확보하기 어렵다.
이를 위해 수치를 이용한 요약 방법이 필요한데, 가장 대표적인 방법이 관측값의 중심위치를 파악하는 **표본평균**(sample mean)이다.

표본평균은 이상치에 민감해서 이상치가 있을 경우 **중앙값**(median)을 사용할 수 있다.

관측값의 중심위치만으로 자료의 분포를 파악하기엔 부족하므로, 관측값이 이 중심위치로부터 얼마나 퍼져 있는지를 나타내는 지표인 **편차**(deviation)도 함께 고려해야 한다.

그러나, 편차의 합은 항상 0이 되므로 편차의 제곱을 사용하여 편차의 크기를 나타낸 **표본분산**(sample variance)을 사용한다.
표본분산은 값을 제곱했기 때문에 원래 단위와 달라지게 된다. 단위를 맞추기 위해 양의 제곱근을 취한 **표본표준편차**(sample standard deviation)를 사용할 수 있다.

이런 중심위치를 나타내는 통계량은 전체 관측값을 크기순으로 정렬했을 때 50%에 해당할 것이다.
이 개념을 확장시켜 (100*p)% 위치에 해당하는 값을 **p백분위수**(percentile)라고 한다. 여기서 p는 위치 비율을 나타내며 0보다 크거나 같고 1보다 작거나 같은 값이다.

특히, 25%, 50%, 75% 위치에 해당하는 값을 각각 제1사분위수(Q1), 제2사분위수(Q2), 제3사분위수(Q3)라고 하며, 이때 Q1에서 Q3까지 전체 자료의 50%를 포함하게 되고 이들 중심엔 중위수가 존재한다.
따라서, 중위수를 중심으로 자료가 떨어진 정도를 알기 위한 측도로 Q1에서 Q3까지의 거리를 이용할 수 있는데, 이를 **사분위수 범위**(interquartile range; IQR)라고 하며 Q3 - Q1로 계산한다.

위에서 살펴본 최소값과 제1사분위수, 중위수, 제3사분위수, 최대값의 다섯가지 요약 통계량을 이용하여 그림으로 나타낼 수 있는데, 이를 **상자 그림**(box plot)이라고 한다.

### 확률변수와 분포
**확률**이란 어떤 실험의 결과에 대해 확신하는 정도를 나타낸 수치적 척도이다.
여기서 **실험**은 출현 가능한 모든 결과들 중에서 오직 한가지 결과만이 나타나는 행위를 의미한다.

출현 가능한 모든 결과를 나타낸 집합을 **표본공간**(sample space)라고 한다. 보통 이 결과 중 특정 결과에 대해 집중한다.
이렇게 표본공간 내에 특정 특성을 나타낸 결과만을 가진 집합을 **사건**(event)이라고 한다.

##### 이산확률변수

표본공간이 커질수록 표본공간을 정의하기 어려워진다. 이때, 표본공간에 속하는 각 결과를 실수에 대응시키는 함수를 **확률변수**(random variable)라고 한다.
예를 들어, 동전 3개를 던졌을 때 앞면이 나오는 횟수를 확률변수로 정의할 수 있다.
- 0: [T, T, T] - 1/8
- 1: [T, T, H], [T, H, T], [H, T, T] - 3/8
- 2: [T, H, H], [H, T, H], [H, H, T] - 3/8
- 3: [H, H, H] - 1/8

확률변수가 가질 수 있는 값에 따라 **이산형 확률변수**(discrete random variable)와 **연속형 확률변수**(continuous random variable)로 나뉜다.
위의 동전 예시는 확률변수가 가질 수 있는 값이 유한하기 때문에 이산형 확률변수이며, 그 값을 가질 확률을 정해주는 규칙 또는 관계를 **확률분포**(probability distribution)라고 한다.
이런 확률변수는 항상 일련의 규칙을 따르기 때문에 함수를 이용한 표현이 가능하며, 이를 **확률분포함수**(probability distribution function)라고 한다.

앞서 관측값의 중심위치를 파악하고 그로부터 퍼진 정도를 파악한 것처럼, 확률변수에서도 그것들을 파악할 수 있다.
이때, 확률변수의 평균을 **기대값**(expected value)이라고 하며, 확률변수의 분산을 **분산**(variance)이라고 한다. 또, 분산의 양의 제곱근을 **표준편차(standard deviation)**라고 한다.
여기서 기대값은 관측할 수 있는 모든 값의 평균이기 때문에 모평균(population mean)과 같고, 통계학에선 이를 μ로 표기한다.

##### 이항분포
이산형 확률변수가 가지는 분포들 중 대표적인 것이 **이항분포**(binomial distribution)이다.
이항분포를 알기 위해선 **베르누이 시행**을 알아야 한다. 베르누이 시행의 특징은 다음과 같다.

1. 각 시행은 성공(Success)과 실패(Failure) 두 가지 결과만을 가질 수 있다.
2. 각 시행은 독립적이다.
3. 하나의 시행에서 성공 확률은 p, 실패 확률은 1-p로 매 시행마다 동일하다.

어느 확률변수가 이항분포를 따르는 것을 확인한다면, 기대값과 분산을 구하는 것은 쉽다.
기대값은 n번 시행 중 성공한 횟수의 기대값이므로 np이고, 분산은 n번 시행 중 성공한 횟수의 분산이므로 np(1-p)이다.

##### 연속형 확률변수
연속형 확률변수는 특정 범위 내에서 어떤 값이든 될 수 있기 때문에 특정 값을 일일이 실수에 대응시키는 것이 불가능하다.
따라서 확률변수가 가질 수 있는 특정 구간에서 확률이 어떻게 분포하는지를 나타내는 **확률밀도함수**(probability density function; pdf)를 사용한다.
그래프 상에서 확률변수가 x축 상의 특정 구간에 속할 확률은 그 구간 아래의 면적으로 나타낼 수 있다. 이 면적의 총 합은 항상 1이다.

이산형 확률변수는 특정값을 가질 확률을 각각 구할 수가 있어 전체를 다 더한 것이고, 연속형 확률변수는 특정 구간에 대한 확률만 정의할 수 있기 때문에 적분을 이용한 것 외엔 기대값과 분산에 대한 의미상의 차이가 없다.
따라서 기대값과 분산에 대한 성질이 동일하다.

##### 정규분포
연속형 확률변수가 가지는 분포들 중 가장 중요한 분포 중 하나가 **정규분포**(normal distribution)이다.
정규분포는 평균 μ와 표준편차 σ로 정의되며, 평균을 중심으로 좌우대칭인 종모양의 분포를 가진다.

정규분포의 중요한 특징 중 하나가 상수를 더하거나 곱해도 정규분포를 유지한다는 것이다.
이 성질을 이용해서 표준화 과정을 통해 평균이 0이고 표준편차가 1인 **표준정규분포**(standard normal distribution)로 변환할 수 있다.
표준화 과정은 (X - μ) / σ로 계산한다.

위에서 본 이항분포의 특징 중 하나는 n이 커질수록 분포의 형태가 점차 대칭에 가까워진 종 모양을 이루게 된다는 것이다.
n이 얼마만큼 커져야할지에 대한 기준은 없지만, np와 np(1-p)가 10 이상이면 이항분포 상에서의 확률과 근사정규분포 상에서의 확률적 차이가 적다고 한다.

### 표집분포와 중심극한정리
모집단의 통계량인 모수를 정확히 아는 것이 불가능하거나 시간과 비용이 많이 들기 때문에 표본을 추출하여 모집단의 특성을 추정한다고 배웠다.
표본으로부터 통계량을 추출할 때마다 값이 변동되는데, 이 변동을 파악할 수 있다면 통계량이 모수와 얼마나 가까운지를 알 수 있을 것이다.
이처럼 통계량은 그 자체가 하나의 확률변수로서 분포를 갖게 되는데, 이를 **표집분포**(sampling distribution)라고 한다.

일반적으로, 임의로 추출한 크기 n의 표본은 서로 독립적이며, 모집단의 분포와 동일한 분포를 가진다고 가정한다.
여기서 표집분포의 중심은 모집단의 평균과 동일하며, 분산과 표준편차는 각각 모집단의 분산과 표준편차에도 영향을 받을 뿐만 아니라, 표본의 크기(n)에도 영향을 받는다.
n이 증가함에 따라 표집분포의 분산과 표준편차는 감소하게 되어, 표집분포의 분포가 모평균을 중심으로 더욱 집중된다.

모집단의 분포가 정규분포라면, 표집분포도 정규분포를 따르게 된다.
모집단의 분포가 정규분포가 아니라면 표집분포는 모집단의 분포에 따라 다르게 나타난다.
하지만, n의 크기가 충분히 크다면(n >= 30) 모집단의 분포와 무관하게 표집분포는 **중심극한정리**(central limit theorem)에 의해 근사적으로 정규분포를 따르게 된다.

### 추정
통계적 추론은 모집단의 수치적 특성인 모수(parameter)를 표본을 통해 추정하는 것을 의미한다.
이 통걔적 추론에는 연구의 목적에 따라 **추정**(estimation)과 **가설검정**(hypothesis testing)으로 나뉘고, 추정에 대해 먼저 살펴보자.

추정은 모수에 대한 추정값을 얻되, 이 추정값의 정밀도를 함께 제시하는 방법으로, **점 추정**(point estimation)과 **구간 추정**(interval estimation)이 있다.

##### 점 추정

점 추정은 단 하나의 값으로 모수를 추정하는 방법이다.
모수를 측정하기 위해 모집단에서 크기 n의 표본을 임의로 추출하고, 이 n개의 확률변수를 이용하여 통계량을 만든 후 표본으로부터 실제 값을 계산하여 하나의 수치를 제시하려고 하는 것이다.
여기서 확률변수로부터 얻은 통계량을 **추정량**(estimator)이라고 하며, 주어진 표본으로부터 계산된 실제 값은 **추정치**(estimate)라고 한다.
추정량은 임의로 추출된 표본에 따라 달라질 수 있기 때문에 표준편차를 계산할 필요가 있다. 이를 **표준오차**(standard error)라고 한다.

##### 구간 추정

구간 추정은 모수의 참값이 포함될 것으로 예상되는 구간을 제시하는 방법이다.
이상적으로 정확하게 참값이 포함될 구간을 제시하는 것이 좋지만, 이는 표본자료의 다양성 때문에 불가능하므로, 제안된 구간이 모수의 참값을 포함할 확률을 명시하게 된다.
이 확률은 대게 90%, 95%로 설정되며 이를 **신뢰수준**(confidence level)이라고 한다. 이 신뢰수준 하에서 추정된 구간은 **신뢰구간**(confidence interval)이라고 부른다.

신뢰구간을 이해하기 위해서 두 가지 경우로 나누어 생각해볼 수 있다.
1. 모집단이 정규분포를 따르고, 모분산을 알고 있는 경우
2. 모집단이 정규분포를 따르고, 모분산을 모르는 보다 현실적인 경우

첫 번째의 경우, 모집단의 분산을 알고 있기 때문에 표준정규분포를 사용하여 신뢰구간을 계산할 수 있다.
두 번째의 경우, 모집단의 분산을 모르기 때문에 표본의 분산을 이용하여 모분산을 추정하고, 이를 표준정규분포로 변환하여 신뢰구간을 계산해야한다.
하지만, 표준화 변환 과정에서 확률변수의 확률분포가 표준정규분포와는 달라지기 때문에 **t분포**(t-distribution)를 사용한다.

t-분포(Student's t-distribution)는 정규분포처럼 0을 중심으로 좌우대칭인 종모양의 확률분포를 갖지만, 꼬리가 더 두꺼운 형태를 띈다.
t-분포의 특징은 자유도가 커질수록 표준정규분포와 근사해간다는 점이다. 통상적으로, 자유도가 30 이상이면 즉, 표본의 크기가 30 이상이면 표준정규분포와 차이가 거의 없다.

### 가설검정

가설검정은 모수에 대한 가설을 세우고, 이 가설이 옳은지 틀린지를 판단하는 방법으로, **귀무가설**(H0, null hypothesis)과 **대립가설**(H1, alternative hypothesis)을 세우고, 표본을 통해 귀무가설을 기각할지 말지를 결정한다.

대립가설은 검증이 필요한 가설로 보통 '효과가 있다', '변화가 있다'의 입장을 나타낸다.
귀무가설은 기본적으로 참이라고 가정되며, 반증될 가설이다. 대립가설과 반대되거나 기존의 주장으로 보통 '효과가 없다', '변화가 없다'의 입장을 나타낸다.

그럼, 어떻게 귀무가설을 기각할지를 결정할까?

예를 들어, '특정 값보다 크면 좋다.'라는 주장을 할 경우 추정량이 클 때 귀무가설을 기각하고 대립가설을 채택할 것이다.
추정량이 클 때를 일반화한 표현으로 *추정량 > c*라고 쓸 수 있는데, 이때 c를 **임계값**(critical value)라고 하며 이 *추정량 > c*라는 구간을 **기각역**(critical region)이라고 한다.

임계값을 구하는 계산은 표본의 크기가 30 이상인 경우, 표준정규분포를 사용하고 이때 표준화된 통계량을 **검정통계량**(test statistic)이라고 한다.
그럼 위의 *추정량 > c* 대신에 *검정통계량 > c*로 바꾸어 생각할 수 있고, 이때 임계값은 귀무가설(H0)를 잘못 기각할 확률의 최대 허용 범위를 나타내는 **유의수준**(significance level)을 고려해야 하고, 이 확률을 작게 하는 것이 이상적이다.
이렇게 검정통계량이 기각역(critical region)에 속할 경우, 다시 말해서 검정통계량이 임계값(critical value)보다 크거나 작을 경우 귀무가설을 기각하고 대립가설을 채택하게 된다.

여기까지 대립가설/귀무가설을 세우고 검정통계량과 유의수준을 고려한 임계값을 설정했다.
그 임계값을 기준으로 귀무가설의 기각 여부를 판단할텐데, 이때 유의수준을 어떻게 설정했는지에 따라서 귀무가설을 잘못 기각할 확률이 달라진다.
따라서 특정 유의수준을 설정함에 따라 귀무가설을 기각할 확률로서 판단하는 방법이 유용할 수 있고, 이를 **유의확률**(significance probability) 즉, **p-value**라고 한다.

가설 검정에서는 두 가지 오류가 발생할 수 있다.
1. **제1종 오류**(Type I error): 귀무가설이 참인데도 기각하는 오류
2. **제2종 오류**(Type II error): 귀무가설이 거짓인데도 채택하는 오류

일반적으로, 제1종 오류가 더 치명적이기 때문에 유의수준을 미리 설정하는 것이다.