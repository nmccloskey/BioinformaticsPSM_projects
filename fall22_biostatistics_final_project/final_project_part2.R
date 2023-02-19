### BIOL 5312 final project part 2


library(naivebayes)
library(tidyverse)
library(ggplot2)
library(psych)

# water potability
# https://www.kaggle.com/datasets/whenamancodes/water-pollution

wp<-read.csv("water_potability.csv")
View(wp)

wp<-drop_na(wp)

xtabs(~Potability,data=wp)
# Potability
# 0    1 
# 1998 1278 

str(wp)
# 'data.frame':	2011 obs. of  10 variables:
#   $ ph             : num  8.32 9.09 5.58 10.22 8.64 ...
# $ Hardness       : num  214 181 188 248 203 ...
# $ Solids         : num  22018 17979 28749 28750 13672 ...
# $ Chloramines    : num  8.06 6.55 7.54 7.51 4.56 ...
# $ Sulfate        : num  357 310 327 394 303 ...
# $ Conductivity   : num  363 398 280 284 475 ...
# $ Organic_carbon : num  18.4 11.6 8.4 13.8 12.4 ...
# $ Trihalomethanes: num  100.3 32 54.9 84.6 62.8 ...
# $ Turbidity      : num  4.63 4.08 2.56 2.67 4.4 ...
# $ Potability     : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...

# convert potability to factor
wp$Potability<-as.factor(wp$Potability)

# training and test data
set.seed(88)
ind<-sample(2,nrow(wp),replace=T,prob=c(0.75,0.25))
X_train<-wp[ind==1,1:9]
y_train<-as.factor(wp[ind==1,10])
X_test<-wp[ind==2,1:9]
y_test<-wp[ind==2,10]

model<-gaussian_naive_bayes(X_train,y_train,usekernel=T)
ptrain<-predict(model,newdata=data.matrix(X_train),type="class")
ctrain<-sum(ptrain==y_train)
(acctrain<-ctrain/length(y_train))
# [1] 0.6211221
ptest<-predict(model,newdata=data.matrix(X_test),type="class")
ctest<-sum(ptest==y_test)
(acctest<-ctest/length(y_test))
# [1] 0.6512097


# disease prediction
# https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning?select=Training.csv
td<-read.csv("Training.csv")
td$prognosis<-as.factor(td$prognosis)
set.seed(88)
ind<-sample(2,nrow(td),replace=T,prob=c(0.75,0.25))
X_train<-td[ind==1,1:132]
y_train<-as.factor(td[ind==1,133])
X_test<-td[ind==2,1:132]
y_test<-td[ind==2,133]

model<-naive_bayes(X_train,y_train,usekernel=T)

ptrain<-predict(model,X_train,type="class")
ctrain<-sum(ptrain==y_train)
(acctrain<-ctrain/length(y_train))
# [1] 0.9991776

ptest<-predict(model,X_test,type="class")
ctest<-sum(ptest==y_test)
(acctest<-ctest/length(y_test))
# [1] 0.9976415
