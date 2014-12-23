---
title: 手写数字识别器
subtitle: 人工神经网络模型
author: htt


license: by-nc-sa
widgets: [mathjax, bootstrap, quiz, shiny, interactive]

url: {lib: ../../libraries}
mode: selfcontained
hitheme: tomorrow
assets: {js: 'test.js'}
ext_widgets : {rCharts: [libraries/nvd3, libraries/highcharts]}


--- 

## 数据和模型

### 模型
人工神经网络（ANN）是一种模仿生物神经网络的结构和功能的数学模型或计算模型。神经网络由大量的人工神经元联结进行计算。大多数情况下人工神经网络能在外界信息的基础上改变内部结构，是一种自适应系统。现代神经网络是一种非线性统计性数据建模工具，常用来对输入和输出间复杂的关系进行建模，或用来探索数据的模式。

### 数据
我们将使用ANN模型进行手写数字识别。我们使用的数据是著名的[MNIST 数据](https://www.kaggle.com/c/digit-recognizer/data)。MNIST数据是一张张28X28像素的手写数字图片。每个像素由一个像素值代表这个像素的明暗程度，更高的数值代表更暗。


--- &twocol

## 可视化部分数据

*** =left

我们使用以下代码可视化部分样本数据

```r
require(graphics)

flip = function(x) {
  xx=matrix(0,nrow(x),ncol(x))
  for (i in (1:nrow(x))){
    xx[i,] = rev(x[i,])}
  return(xx)}

sample_plot = function(x,n) {
  xx = list()
  par(mfrow=c(sqrt(n), sqrt(n)),mar=rep(0.2,4))
  for (i in 1:n) {
    temp = as.numeric(x[i,])
    temp = matrix(temp,28,28)
    xx[[i]] = flip(temp)
    image(z=xx[[i]], col=gray.colors(12),xaxt='n',yaxt='n',ann=FALSE)
  }}

sample_p <- read.csv("data\train.csv", header=TRUE,nrows=100)
sample_p = sample_p[,-1]
sample_plot(sample_p,16)
```


*** =right

![plot of chunk unnamed-chunk-2](assets/fig/unnamed-chunk-2-1.png) 


--- &twocol

## ANN的向前传播


我们使用单层ANN模型。右上角括号内的数字代表变量所在的层，在本模型（单层ANN模型）中0代表输入层，1代表隐藏层，2代表输出层。$X^{(0)}$ 是我们的原始数据（手写数字图片），一行784（28X28）列的数据。$f_{\theta}(a)$ 是一个sigmoid函数。$a$ 是隐藏层的变量，$z$ 是通过sigmoid函数后的隐藏层变量。$\Theta^{(l)}_0$ 是输入层和隐藏层的偏项。$\widehat{Y}$ 是我们的预测值，一个数值介于0到1之间的一行十列的矩阵。当这矩阵中最大值处在第三列时，我们对这图片的预测结果便是3。

***=left

$$
a^{(1)} = \Theta^{(1)}_0+X^{(0)}(\Theta^{(1)})^T
$$

$$
z^{(1)} = f_{\theta}(a^{(1)})
$$

$$
a^{(2)} = \Theta^{(2)}_0+z^{(1)}(\Theta^{(2)})^T
$$

$$
\hat{Y} = f_{\theta}(a^{(2)})
$$




***=right

相关代码:


```r
  a1 = cbind(1,data)
  z2 = a1%*%t(Theta1)
  a2 = cbind(1,sigmoid(z2))
  z3 = a2%*%t(Theta2)
  h = sigmoid(z3)
  
  ny = matrix(0, length(y),num_labels)
  for (i in 1:length(y)){
    ny[i,y[i]] = 1
  }
```

---

## ANN的成本函数

对于分类问题，我们使用Cross entropy成本函数，这是模型的成本函数：

$$
J(\Theta)= -\frac{1}{m}  \left[  \sum^m_i \sum^{10}_k y^{(i)}_{k}log f_\theta (z^{(i)})_{k}+ (1-y^{(i)}_k ) log( 1- f_\theta (z^{(i)})_{k} )  \right]
$$


或简化版：

$$
J(\Theta)= -\frac{1}{m} \sum^m_i \sum^{10}_k y^{(i)}_{k}log f_\theta (z^{(i)})_{k}      
$$

我们的目标就是通过改变 $\Theta$ 来最小化成本函数 $J(\Theta)$ .

相关代码：

```r
  regu = lambda*(sum(Theta1[,-1]^2)+sum(Theta2[,-1]^2))/(2*m)
  cost = -sum(ny*log(h)+(1-ny)*log(1-h))/m+regu
```


--- &twocol

## ANN的反向传播
根据成本函数，我们可以得到 $\Theta^{(1)}$ 和 $\Theta^{(2)}$ 的导数。得到导数之后，使用梯度下降法来求出 $J(\Theta)$ 的本地最小值。

***=left

公式推导：
$$
\frac{\partial J(\Theta)}{\partial \Theta^{(2)}} =  \frac{\partial f_{\theta}(a^{(2)})}{\partial a^{(2)}} \frac{\partial a^{(2)}}{\partial \Theta^{(2)}}
$$

$$
\frac{\partial J(\Theta)}{\partial \Theta^{(2)}} =  (y-f_{\theta}(a^{(2)}))z^{(1)}
$$

$$
\frac{\partial J(\Theta)}{\partial \Theta^{(1)}} =  \frac{\partial f_{\theta}(a^{(2)})}{\partial a^{(2)}} \frac{\partial a^{(2)}}{\partial \Theta^{(2)}} \frac{\partial f_{\theta}(a^{(1)})}{\partial a^{(1)}} \frac{\partial a^{(1)}}{\partial \Theta^{(1)}}
$$

$$
\frac{\partial J(\Theta)}{\partial \Theta^{(1)}} = \frac{\partial J(\Theta)}{\partial \Theta^{(2)}} \frac{\partial f_{\theta}(a^{(1)})}{\partial a^{(1)}} X^{(0)}
$$



***=right
相关代码：

```r
  delta3 = h-ny
  delta2 = (delta3%*%Theta2[,-1])*sigmoidGradient(z2)
  thres1 = matrix(1,nrow(Theta1),ncol(Theta1))
  thres1[,1] = 0
  thres2 = matrix(1,nrow(Theta2),ncol(Theta2))
  thres2[,1] = 0
  Theta1_grad = (t(delta2)%*%a1)/m + thres1*Theta1*lambda/m
  Theta2_grad = (t(delta3)%*%a2)/m + thres2*Theta2*lambda/m
```


--- &twocol

## ANN的R代码实现

*** =left

将之前三部分合并起来就是ANN的主要实现代码。

右边这个函数输入变量包括：data 、 y、  h_layer、  lambda、  num_labels
 - data是训练数据矩阵，每一行代表一个数据。
 - y是训练数据的真确结果，每一行代表一个数据。
 - h_layer是隐藏层的变量个数，默认15。
 - num_labels是输出结果的变量个数，或者数据的类别数。
 - lambda是过拟合控制变量。

输出变量包括：cost、 h、 grad
 - cost是对应的成本函数值。
 - h是对应输出层的值。
 - grad是$\Theta^{(1)}$ 和 $\Theta^{(2)}$ 的导数，用于优化成本函数。
*** =right


```r
NN_cost = function(data, y, init, h_layer=15, lambda=0, num_labels=10) {
  m = nrow(data)
  # reshape Theta1 and Theta2
  in_layer = ncol(data)
  par_t1 = init[c(1:((in_layer+1)*h_layer))]

  par_t2 = init[c(((in_layer+1)*h_layer+1):length(init))]
  Theta1 = matrix(par_t1,h_layer,in_layer+1)
  Theta2 = matrix(par_t2,num_labels,h_layer+1)
  # forward propagation
  a1 = cbind(1,data)
  z2 = a1%*%t(Theta1)
  a2 = cbind(1,sigmoid(z2))
  z3 = a2%*%t(Theta2)
  h = sigmoid(z3)
  
  ny = matrix(0, length(y),num_labels)
  for (i in 1:length(y)){
    ny[i,y[i]] = 1
  }
  
  cost = -sum(ny*log(h)+(1-ny)*log(1-h))/m+lambda*(sum(Theta1[,-1]^2)+sum(Theta2[,-1]^2))/(2*m)
  # back propagation
  delta3 = h-ny
  delta2 = (delta3%*%Theta2[,-1])*sigmoidGradient(z2)
  thres1 = matrix(1,nrow(Theta1),ncol(Theta1))
  thres1[,1] = 0
  thres2 = matrix(1,nrow(Theta2),ncol(Theta2))
  thres2[,1] = 0
  Theta1_grad = (t(delta2)%*%a1)/m + thres1*Theta1*lambda/m
  Theta2_grad = (t(delta3)%*%a2)/m + thres2*Theta2*lambda/m
  
  result = list(cost=cost,h=h,grad=list(t1=Theta1_grad,t2=Theta2_grad))
  return(result)
}
```







--- &twocol

## ANN模型训练

***=left

 - 首先我们随机选取出70%的数据作为训练数据



 - 对训练数据进行预处理和归一化

    
    
    
 - 建立训练函数和起始值

 - 最后使用fmincg函数对模型进行最优化，因为fmincg比我自己写的梯度下降法效率更高。fmincg是Andrew Ng在machine learning在线课程里面使用的优化算法之一，由matlab写成，我使用R重新将它写一遍。具体代码请见[fmincg](   )




***=right

```r
data_all <- read.csv("train.csv", header=TRUE)
train_size = floor(0.7*nrow(data_all))
train_indx = sample(seq_len(nrow(data_all)), size = train_size)
```


```r
data = data_all[train_indx,]
train = data[,-1]
train = data.matrix(train)
y = data$label
y=replace(y,y==0,10)
train_nl = (train-125)/255
```


```r
h_layer=25
t1_epsilon = sqrt(6)/sqrt(ncol(train)+h_layer)
t2_epsilon = sqrt(6)/sqrt(h_layer+10)
t1_random = matrix(runif((ncol(train)+1)*25,0,1),25,(ncol(train)+1))
t2_random = matrix(runif(26*10,0,1),10,26)
t1_random = t1_random*2*t1_epsilon - t1_epsilon
t2_random = t2_random*2*t2_epsilon - t2_epsilon
ini = c(as.vector(t1_random),as.vector(t2_random))
lambda=1
costFunction = function(p) {
  result = NN_cost(train_nl,y,p,lambda=lambda)
  J = result$cost
  grad = c(as.vector(result$grad$t1),as.vector(result$grad$t2))
  grad = as.matrix(grad,length(grad),1)
  return(list(J=J,grad=grad))
}
```


```r
optimization = fmincg(f=costFunction, X=par, Maxiter=500)
```




---

## 结果检测

优化完成后，可以用以下函数测试优化结果。
 - t就是最小化成本函数的$\Theta^{(1)}$ 和 $\Theta^{(2)}$ 
 - data是检测的数据
 - y是正确输出值

```r
check_accurate = function(data, y, t, h_layer=25,lambda,num_labels=10) {
  fit_model = NN_cost(data,y,init=t,h_layer,lambda=lambda,num_labels)
  fit_matirx = fit_model$h
  fit_values = apply(fit_matirx,1,which.max)
  fit_values = as.vector(fit_values)
  y = as.vector(y)
  accurate = sum(fit_values==y)/length(y)
  result = list(fit = fit_values, acu = accurate)
  return(result)
}
```


将剩余数据分为测试数据和验证数据


```r
test_valid_data = data_all[-train_indx,]
test_size = floor(0.7*nrow(test_valid_data))
test_ind = sample(seq_len(nrow(test_valid_data)), size = test_size)
test_data = test_valid_data[test_ind,]
valid_data = test_valid_data[-test_ind,]
```


---

## 结果检测


分别对测试数据和验证数据进行处理


```r
#### set up test set
test = test_data[,-1]
test = data.matrix(test)
test_y = test_data$label
test_y = replace(test_y,test_y==0,10)

test_nl = (test-125)/255

test_accurate = check_accurate(test_nl,test_y,par,lambda=lambda)
test_accurate$acu

#### set up valid set
valid = valid_data[,-1]
valid = data.matrix(valid)
valid_y = valid_data$label
valid_y = replace(valid_y,valid_y==0,10)

valid_nl = (valid-125)/255

valid_accurate = check_accurate(valid_nl,valid_y,par,lambda=lambda)
valid_accurate$acu
```



