---
title: "Peer-graded Assignment: Prediction Assignment"
author: "Ivo Giulietti"
date: "27/12/2021"
output:
  html_document:
    self_contained: true
    keep_md: yes
---



## Data Prep

First we download the data from the URLs provided. Also we load the initial packages we are going to use. 


```r
library(tidyverse)
library(caret)
library(lubridate)
library(rattle)
library(DataExplorer)
library(doParallel)

url.train='https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
url.test='https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
```

We load the data,run a quick EDA and realize a lot of missing values. More than 95% in 97 columns. These columns might not be useful for our models. So we decided to  


```r
download.file(url.train,'train.csv',mode = 'wb')
#First row was imported with index numbers. I remove it 
train=read_csv('train.csv',lazy=FALSE) %>% select(-1) 

#DataExplorer::create_report(train)

download.file(url.test,'test.csv')
#First row was imported with index numbers. I remove it 
test=read_csv('test.csv',lazy=FALSE)%>% select(-1) 

dim(train)
```

```
## [1] 19622   159
```

```r
dim(test)
```

```
## [1]  20 159
```

```r
train=train %>% 
        mutate(across(.cols = contains('arm|belt|dumbbell|forearm'),.fns = ~as.numeric))

missing_values=apply(train,2,function(x){length(which(is.na(x)))})

num_col=length(which(missing_values>0.9))

percentage.missing=missing_values/nrow(train)


train=train%>% select(-names(missing_values)[missing_values>0.9]) %>% mutate(classe=as.factor(classe))
```

## Model

### Data Partition & Cluster


```r
nucleos=detectCores()-1
cl <- makeCluster(nucleos)
registerDoParallel(cl)

set.seed(12486)
#Creo datasets de training y testing
inTrain = createDataPartition(y=train$classe,
                              p=0.8,list=FALSE)

training=train[inTrain,]
testing=train[-inTrain,]
```


### Decision Tree

First option is to see how a decision tree performs. We will perform a BoxCox preprocess to obtain better results. We also tried without preprocessing, center scale and pca analysis. Nevertheless BoxCox performed the best. 


```r
set.seed(45687)
fit.dt = train(classe ~ .,
              method = 'rpart',
              data = training,
              na.action=na.exclude,
              preProces=c('BoxCox'))

fit.dt
```

```
## CART 
## 
## 15699 samples
##    58 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: Box-Cox transformation (5) 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 15699, 15699, 15699, 15699, 15699, 15699, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03836226  0.5471724  0.41014397
##   0.04223409  0.5110387  0.35772908
##   0.11562083  0.3230845  0.05847338
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.03836226.
```

```r
fancyRpartPlot(fit.dt$finalModel)
```

![](peer-graded-assignment-Ivo-_files/figure-html/decision tree-1.png)<!-- -->

```r
preddt=predict(fit.dt,newdata=testing)
confusionMatrix(preddt,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 994 242 182 205  14
##          B  27 256  25 138  27
##          C  82 261 476 135  17
##          D   0   0   0   0   0
##          E  13   0   1 165 663
## 
## Overall Statistics
##                                           
##                Accuracy : 0.609           
##                  95% CI : (0.5935, 0.6243)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4957          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8907  0.33729   0.6959   0.0000   0.9196
## Specificity            0.7709  0.93142   0.8472   1.0000   0.9441
## Pos Pred Value         0.6072  0.54123   0.4902      NaN   0.7874
## Neg Pred Value         0.9466  0.85420   0.9295   0.8361   0.9812
## Prevalence             0.2845  0.19347   0.1744   0.1639   0.1838
## Detection Rate         0.2534  0.06526   0.1213   0.0000   0.1690
## Detection Prevalence   0.4173  0.12057   0.2475   0.0000   0.2146
## Balanced Accuracy      0.8308  0.63435   0.7715   0.5000   0.9318
```

 This model doesnt seem to have a high accuracy. 
 
### Random Forest 


```r
set.seed(45687)
fit.rf = train(classe ~ .,
              method = 'rf',
              data = training,
              na.action=na.exclude,
              preProces=c('BoxCox'))

saveRDS(object = fit.rf,file = 'fit_rf.rsd')
```



```r
fit.rf=readRDS('fit_rf.rsd')

tiempo=fit.rf$times$everything['elapsed']
tiempo.final=fit.rf$times$final['elapsed']



plot(fit.rf)
```

![](peer-graded-assignment-Ivo-_files/figure-html/random forest testing-1.png)<!-- -->

```r
model.accuracy=postResample(pred = preddt, obs = testing$classe)['Accuracy']


preddt=predict(fit.rf,newdata=testing)

confusionMatrix(preddt,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  759    0    0    0
##          C    0    0  684    2    0
##          D    0    0    0  641    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9995          
##                  95% CI : (0.9982, 0.9999)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9994          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   0.9969   1.0000
## Specificity            1.0000   1.0000   0.9994   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   0.9971   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   0.9994   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.1744   0.1634   0.1838
## Detection Prevalence   0.2845   0.1935   0.1749   0.1634   0.1838
## Balanced Accuracy      1.0000   1.0000   0.9997   0.9984   1.0000
```

```r
col_index <- varImp(fit.rf)$importance %>% 
  mutate(names=row.names(.)) %>%
  arrange(-Overall)
```

Training this model without any further limitation on cross validation or the length of the tuning took 16.514 minutes to train. The final model took 0.8133333 minutes. If we have limited resources or time, we could train the model with less than 2 variables and we would get results of +98% as the graph shows. The optimum is about 40 Predictors. 

The accuracy of the model is formidable. Using the testing set we got an accuracy of  **99.94%** . Since we've got an almost perfect result on both training and testing we will not try other models. The training data set has been cross validated and everything appears to be in order. 

### Quiz 


Quiz resulted in 100% . 

```r
preddt=predict(fit.rf,newdata=test)

preddt
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

