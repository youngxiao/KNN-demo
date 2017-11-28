# KNN simple demo
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




<div align=center><img height="320" src="https://github.com/youngxiao/Linear-Regression-demo/raw/master/result/2D_regression.png"/></div>


## Section 2: 3D 线性回归
首先，对二氧化碳与气候变化数据集 `global_co2.csv` `annul_temp.csv`进行预处理，分别保留两个数据 1960 年以后的数据，得到 
[Year, CO2 emissions], [Year, Global temperature] 两个数据集，然后合并，变为 [Year, CO2 emissions, Global temperature]
```
# Import data
co2_df = pd.read_csv('global_co2.csv')
temp_df = pd.read_csv('annual_temp.csv')
print(co2_df.head())
print(temp_df.head())

# Clean data
co2_df = co2_df.ix[:,:2]                     # Keep only total CO2
co2_df = co2_df.ix[co2_df['Year'] >= 1960]   # Keep only 1960 - 2010
co2_df.columns=['Year','CO2']                # Rename columns
co2_df = co2_df.reset_index(drop=True)                # Reset index

temp_df = temp_df[temp_df.Source != 'GISTEMP']                              # Keep only one source
temp_df.drop('Source', inplace=True, axis=1)                                # Drop name of source
temp_df = temp_df.reindex(index=temp_df.index[::-1])                        # Reset index
temp_df = temp_df.ix[temp_df['Year'] >= 1960].ix[temp_df['Year'] <= 2010]   # Keep only 1960 - 2010
temp_df.columns=['Year','Temperature']                                      # Rename columns
temp_df = temp_df.reset_index(drop=True)                                             # Reset index

print(co2_df.head())
print(temp_df.head())

# Concatenate
climate_change_df = pd.concat([co2_df, temp_df.Temperature], axis=1)

print(climate_change_df.head())
```

显示三维可视化数据
```
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
fig.set_size_inches(12.5, 7.5)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs=climate_change_df['Year'], ys=climate_change_df['Temperature'], zs=climate_change_df['CO2'])

ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
ax.view_init(10, -45)
```
<div align=center><img height="320" src="https://github.com/youngxiao/Linear-Regression-demo/raw/master/result/3D_data.png"/></div>

将二氧化碳排放和全球温度变化分别用二维显示
```
f, axarr = plt.subplots(2, sharex=True)
f.set_size_inches(12.5, 7.5)
axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
axarr[0].set_ylabel('CO2 Emissions')
axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
axarr[1].set_xlabel('Year')
axarr[1].set_ylabel('Relative temperature')
```
<div align=center><img width="600" src="https://github.com/youngxiao/Linear-Regression-demo/raw/master/result/co2.png"/></div>
<div align=center><img width="600" src="https://github.com/youngxiao/Linear-Regression-demo/raw/master/result/temp.png"/></div>

3D线性回归并可视化结果
```
X = climate_change_df.as_matrix(['Year'])
Y = climate_change_df.as_matrix(['CO2', 'Temperature']).astype('float32')
X_train, X_test, y_train, y_test = np.asarray(train_test_split(X, Y, test_size=0.1))
reg = LinearRegression()
reg.fit(X_train, y_train)
print('Score: ', reg.score(X_test.reshape(-1, 1), y_test))
x_line = np.arange(1960,2011).reshape(-1,1)
p = reg.predict(x_line).T
fig2 = plt.figure()
fig2.set_size_inches(12.5, 7.5)
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(xs=climate_change_df['Year'], ys=climate_change_df['Temperature'], zs=climate_change_df['CO2'])
ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
ax.plot(xs=x_line, ys=p[1], zs=p[0], color='green')
ax.view_init(10, -45)
```
<div align=center><img height="320" src="https://github.com/youngxiao/Linear-Regression-demo/raw/master/result/3D_regression.png"/></div>

将对二氧化碳和全球气温变化的预测分别在二维里面显示
```
f, axarr = plt.subplots(2, sharex=True)
f.set_size_inches(12.5, 7.5)
axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
axarr[0].plot(x_line, p[0])
axarr[0].set_ylabel('CO2 Emissions')
axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
axarr[1].plot(x_line, p[1])
axarr[1].set_xlabel('Year')
axarr[1].set_ylabel('Relative temperature')
```
<div align=center><img width="600" src="https://github.com/youngxiao/Linear-Regression-demo/raw/master/result/co2.png"/></div>
<div align=center><img width="600" src="https://github.com/youngxiao/Linear-Regression-demo/raw/master/result/temp.png"/></div>

## 依赖的 packages
* matplotlib
* pandas
* numpy
* seaborn

## 欢迎关注
* Github：https://github.com/youngxiao
* Email： yxiao2048@gmail.com  or yxiao2017@163.com
* Blog:   https://youngxiao.github.io
