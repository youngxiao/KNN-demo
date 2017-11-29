# KNN simple demo

k-近邻（KNN）算法可用于分类或回归。这是一种简单的机器学习算法，但它通常非常有效。它是非参数的，也就是说它没有固定数量的参数。当你训练KNN时，它找到k个“最近点”到一个给定的点，并返回最高比例的类。如果k＝1，则只查找一个最接近的点并返回它的类。这不太理想。KNN的最优K值通常在3-10。测试输入x与训练样本之间的距离欧氏距离是典型的距离度量。其他的距离度量也被使用，但这是最常见的。当我们拥有大量的特性时，维数灾难使得KNN失效了。特别是嘈杂的或不相关的特征。从某种意义上说，噪音使两个点彼此距离越近。可以使用诸如PCA之类的工具来减少维度，如果有超过10个特征，这是一个很好的实践。下面是利用[Breast Cancer Wisconsin (Diagnostic) Database](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) 数据集来训练一个分类器，用于病人的诊断。同样还是用到机器学习库 [sklearn](http://scikit-learn.org/stable/install.html#install-bleeding-edge%E3%80%82).


### 1. 导入数据
在本示例中将用到 [Breast Cancer Wisconsin (Diagnostic) Database](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) 数据集来训练一个分类器，用于病人的诊断. 因此先导入数据集，并看一下数据集的描述.
```
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.DESCR) # 打印数据集的描述
cancer.keys()       # 数据集关键字
```

| variables(mean)                    |   Min |   Max |
| ---------------------------------- |:-----:| -----:|
|radius (mean):                      |  6.981|  28.11|
|texture (mean):                     |  9.71 |  39.28|
|perimeter (mean):                   |  43.79|  188.5|
|area (mean):                        |  143.5|  2501.0|
|smoothness (mean):                  |  0.053|  0.163|
|compactness (mean):                 |  0.019|  0.345|
|concavity (mean):                   |  0.0  |  0.427|
|concave points (mean):              |  0.0  |  0.201|
|symmetry (mean):                    |  0.106|  0.304|
|fractal dimension (mean):           |  0.05 |  0.097|
|radius (standard error):            |  0.112|  2.873|
|texture (standard error):           |  0.36 |  4.885|
|perimeter (standard error):         |  0.757|  21.98|
|area (standard error):              |  6.802|  542.2|
|smoothness (standard error):        |  0.002|  0.031|
|compactness (standard error):       |  0.002|  0.135|
|concavity (standard error):         |  0.0  |  0.396|
|concave points (standard error):    |  0.0  |  0.053|
|symmetry (standard error):          |  0.008|  0.079|
|fractal dimension (standard error): |  0.001|  0.03|
|radius (worst):                     |  7.93 |  36.04|
|texture (worst):                    |  12.02|  49.54|
|perimeter (worst):                  |  50.41|  251.2|
|area (worst):                       |  185.2|  4254.0|
|smoothness (worst):                 |  0.071|  0.223|
|compactness (worst):                |  0.027|  1.058|
|concavity (worst):                  |  0.0  |  1.252|
|concave points (worst):             |  0.0  |  0.291|
|symmetry (worst):                   |  0.156|  0.664|
|fractal dimension (worst):          |  0.055|  0.208|

### 2. 特征数量

乳腺癌数据集有多少个特征? 下面的函数返回一个整数值
```
def answer_zero():
    return len(cancer['feature_names'])

answer_zero() #调用
```

### 3. 数据集转换成 DataFrame

Scikit-learn 通常和 lists, numpy arrays, scipy-sparse matrices 以及 pandas DataFrames 一起使用，下面就用 pandas 将数据集转换成 DataFrame，这样对数据各种操作更方便. 

下面的函数返回一个 `(569, 31)` 的 DataFrame ，并且 columns 和 index 分别为

    
```
def answer_one():
    
    df = pd.DataFrame( data = cancer['data'], index=range(0,569), columns=  ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension'])
    df['target'] = cancer['target']

    
    return df

answer_one()
```

### 4. 类别分布
数据集样本的类别的分布是怎样呢? ，例如有多少样本是 `malignant`(恶性) (编码为 0)，有多少样本是`benign`(良性) (编码为 1)?

下面的函数返回 恶性 和 良性 两个类别各样本数 index = `['malignant', 'benign']`
```
def answer_two():
    cancerdf = answer_one()
    
    yes = np.sum([cancerdf['target'] > 0])
    no = np.sum([cancerdf['target'] < 1])
    
    data = np.array([no, yes])
    s = pd.Series(data,index=['malignant','benign'])
    
    return s

answer_two()
```


### 5. 数据准备1
将上面的 DataFrame 分为 `X` (data，特征) and `y` (label，标签).

*下面函数返回:* `(X, y)`*, * 
* `X` *has shape* `(569, 30)` 30个特征
* `y` *has shape* `(569,)`.


```
from sklearn.model_selection import train_test_split
def answer_three():
    cancerdf = answer_one()
    
    X = cancerdf[  ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension'] ]
    y = cancerdf['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X, y
```

### 6. 数据准备2
利用 `train_test_split`, 将 `X`，`y` 分别分到训练集和测试集 `(X_train, X_test, y_train, and y_test)`.

**设置随机数为 0，`random_state=0` **

*下面函数返回；* `(X_train, X_test, y_train, y_test)`
* `X_train` *has shape* `(426, 30)`
* `X_test` *has shape* `(143, 30)`
* `y_train` *has shape* `(426,)`
* `y_test` *has shape* `(143,)`

```
from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test
```

### 7. sklearn 中的 KNN 分类器
利用 KNeighborsClassifier, 以及训练集 `X_train`, `y_train` 来训练模型，并设置 `n_neighbors = 1`.
```
from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    
    return knn
```

### 8. 模型预测
利用 KNN 分类模型，以及每个的特征的均值来预测样本.

`cancerdf.mean()[:-1].values.reshape(1, -1)` 可以获得每个特征的均值, 忽略 target, 并 reshapes 数据从 1D 到 2D.

*下面的函数返回一个数组 `array([ 0.])` or `array([ 1.])`*

```
from sklearn.neighbors import KNeighborsClassifier
def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    A = answer_five()
    prediction = A.predict(means)

    return prediction

answer_six()
```

### 9. 测试集预测
利用上面的 KNN 模型预测数据 `X_test`的类别 y.

*下面函数返回一个大小为 `(143,)` 的数组，并且值是 `0.0` or `1.0`.*

```
def answer_seven():
    cancerdf = answer_one()
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    prediction =  knn.predict(X_test)
    
    return prediction

answer_seven()
```

### 10. 模型分数
现在利用 `X_test` 和 `y_test`来看看模型能得多少分（平均准确度），原本的测试集就有label标签，而利用 KNN 同样可以对测试集预测得到另一组标签，从而计算出模型的平均准确度.

*下面函数返回一个小数，0-1*

```
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score = (knn.score(X_test, y_test))
    
    return score

answer_eight()
```

### 11. plot

同样可以看一下在模型的作用下，训练集 和 测试集中恶性样本和良性样本分别的准确度，训练集上都是 100％，而在测试集上，良性样本的准确度略高.

```
def accuracy_plot():
    import matplotlib.pyplot as plt

    %matplotlib notebook

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    
# Uncomment the plotting function to see the visualization, 
# Comment out the plotting function when submitting your notebook for grading

accuracy_plot()
```

<div align=center><img height="320" src="https://github.com/youngxiao/KNN-demo/raw/master/images/Acurracy.png"/></div>



## 依赖的 packages
* sklearn
* pandas
* numpy

## 欢迎关注
* Github：https://github.com/youngxiao
* Email： yxiao2048@gmail.com  or yxiao2017@163.com
* Blog:   https://youngxiao.github.io
