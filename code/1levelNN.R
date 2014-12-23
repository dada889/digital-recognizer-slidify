setwd("/media/htt/Ha/MOOC/Kaggle/Digit Recognizer")



#data <- read.csv("train.csv", header=TRUE, nrows=1000)
data_all <- read.csv("train.csv", header=TRUE)
train_size = floor(0.7*nrow(data_all))
train_indx = sample(seq_len(nrow(data_all)), size = train_size)


data = data_all[train_indx,]
train = data[,-1]
train = data.matrix(train)
y = data$label
y=replace(y,y==0,10)



####################################################################################
##################        function set up            ###############################
####################################################################################

sigmoid = function(x) {
  
  return(1/(1+exp(-x)))
}

sigmoidGradient = function(x) {
  g = exp(-x)/((1+exp(-x))^2)
  return(g)}




# input training data should be 1 line 1 sample
# output y should be single line or column of data from 1 to 10, replace 0 by 10.
# This function taking initial value as vector
NN_cost = function(data, y, init, h_layer=25, lambda=0, num_labels=10) {
  
  m = nrow(data)
  # reshape Theta1 and Theta2
  in_layer = ncol(data)
  par_t1 = init[c(1:((in_layer+1)*h_layer))]

  par_t2 = init[c(((in_layer+1)*h_layer+1):length(init))]
  Theta1 = matrix(par_t1,h_layer,in_layer+1)
  Theta2 = matrix(par_t2,num_labels,h_layer+1)
  
  
  a1 = cbind(1,data)
  z2 = a1%*%t(Theta1)
  a2 = cbind(1,sigmoid(z2))
  z3 = a2%*%t(Theta2)
  h = sigmoid(z3)
  
  ny = matrix(0, length(y),num_labels)
  for (i in 1:length(y)){
    ny[i,y[i]] = 1
  }
  
  regu = lambda*(sum(Theta1[,-1]^2)+sum(Theta2[,-1]^2))/(2*m)
  cost = -sum(ny*log(h)+(1-ny)*log(1-h))/m+regu
  
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


# take a vector of initial value
# return a cost value and a column of gradient
costFunction = function(p) {
  result = NN_cost(train255,y,p,lambda=lambda)
  J = result$cost
  grad = c(as.vector(result$grad$t1),as.vector(result$grad$t2))
  grad = as.matrix(grad,length(grad),1)
  return(list(J=J,grad=grad))
}



# f is a function take in only one argument, a column of initial value
# X is one single column matirx initial value
fmincg = function(f,X, Maxiter=10) {
  length = Maxiter
  RHO = 0.01 
  SIG = 0.5  
  INT = 0.1  
  EXT = 3.0  
  MAX = 20   
  RATIO = 100
  
  red = 1
  
  i = 0
  ls_failed = 0
  fX = numeric(0)
  eval = f(X)
  f1 = eval$J
  df1 = eval$grad
  i = i + (length<0)
  s = -df1
  d1 = t(-s)%*%s
  z1 = red/(1-d1)
  
  while (i < abs(length)) {
    i = i + (length>0)
    
    X0 = X; f0 = f1; df0 = df1
    X = X + z1[1]*s
    eval = f(X)
    f2 = eval$J
    df2 = eval$grad
    i = i + (length<0)
    d2 = t(df2)%*%s;
    f3 = f1; d3 = d1; z3 = -z1;
    if(length>0){M=MAX}else{M=min(MAX,-length-i)}
    success = 0; limit = -1; 
    while(1){
      while(((f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1)) & (M > 0)) {
        limit = z1
        if (f2 > f1){
          z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)
        } else {
          A = 6*(f2-f3)/z3+3*(d2+d3)                                
          B = 3*(f3-f2)-z3*(d3+2*d2)
          z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A}
        if (is.na(z2)|is.infinite(z2)){z2 = z3/2}
        z2 = max(min(z2, INT*z3),(1-INT)*z3)
        z1 = z1 + z2
        X = X + z2[1]*s
        eval = f(X)
        f2 = eval$J
        df2 = eval$grad
        M = M - 1; i = i + (length<0)
        d2 = t(df2)%*%s
        z3 = z3 - z2
      }
      if (f2 > (f1+z1*RHO*d1) | d2 > -SIG*d1) {
        break
      } else if (d2 > SIG*d1) {
        success = 1; break
      } else if (M ==0) {
        break
      }
      A = 6*(f2-f3)/z3+3*(d2+d3)
      B = 3*(f3-f2)-z3*(d3+2*d2)
      z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3))
      if (is.na(z2)|is.infinite(z2)|z2<0) {
        if (limit <-0.5) {
          z2 = z1*(EXT-1)
        } else {
          z2 = (limit-z1)/2
        }
      } else if ((limit > -0.5) & ((z2+z1) > limit)){
        z2 = (limit-z1)/2
      } else if ((limit < -0.5) & ((z2+z1) > z1*EXT)) {
        z2 = z1*(EXT-1)
      } else if (z2 < (-z3*INT)) {
        z2 = -z3*INT
      } else if ((limit > -0.5) & (z2 < (limit-z1)*(1.0-INT))) {
        z2 = (limit - z1)*(1 - INT)
      }
      f3 = f2; d3 = d2; z3 = -z2; 
      z1 = z1 + z2; X = X + z2[1]*s;
      eval = f(X)
      f2 = eval$J
      df2 = eval$grad
      M = M - 1; i = i + (length<0)
      d2 = t(df2)%*%s
    }
    
    if (success) {
      f1 = f2; fX = rbind(fX, f1)
      print(cat('Iteration  ',i,'| Cost:',f1))
      temp = (t(df2)%*%df2-t(df1)%*%df2)/(t(df1)%*%df1)
      s = temp[1]*s - df2
      tmp = df1; df1 = df2; df2 = tmp;
      d2 = t(df1)%*%s
      if (d2>0) {
        s = -df1
        d2 = -t(s)%*%s
      }
      z1 = z1 * min(RATIO, d1/(d2-2.2251e-308))
      d1 = d2
      ls_failed = 0
    } else {
      X = X0; f1 = f0; df1 = df0;
      if (ls_failed|i > abs(length)) {
        break
      }
      tmp = df1; df1 = df2; df2 = tmp;
      s = -df1
      d1 = -t(s)%*%s
      z1 = 1/(1-d1)
      ls_failed = 1
    }
  }
  return(list(par=X,cost=fX))
}




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

####################################################################################
##################        Train the data             ###############################
####################################################################################

h_layer=25
t1_epsilon = sqrt(6)/sqrt(ncol(train)+h_layer)
t2_epsilon = sqrt(6)/sqrt(h_layer+10)

t1_random = matrix(runif((ncol(train)+1)*25,0,1),25,(ncol(train)+1))
t2_random = matrix(runif(26*10,0,1),10,26)
t1_random = t1_random*2*t1_epsilon - t1_epsilon
t2_random = t2_random*2*t2_epsilon - t2_epsilon
ini = c(as.vector(t1_random),as.vector(t2_random))
lambda=1
train_nl = (train-125)/255


costFunction = function(p) {
  result = NN_cost(train_nl,y,p,lambda=lambda)
  J = result$cost
  grad = c(as.vector(result$grad$t1),as.vector(result$grad$t2))
  grad = as.matrix(grad,length(grad),1)
  return(list(J=J,grad=grad))
}

first_try = fmincg(f=costFunction, X=par, Maxiter=500)




par = first_try$par
write.csv(file="98percent_train.csv",par) # only read the second column


temp_accurate = check_accurate(train_nl,y,par,lambda=lambda)
temp_accurate$acu

####################################################################################
##################        test and valid             ###############################
####################################################################################


#### split the rest of data into test set and valid set
test_valid_data = data_all[-train_indx,]
test_size = floor(0.7*nrow(test_valid_data))
test_ind = sample(seq_len(nrow(test_valid_data)), size = test_size)
test_data = test_valid_data[test_ind,]
valid_data = test_valid_data[-test_ind,]

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








