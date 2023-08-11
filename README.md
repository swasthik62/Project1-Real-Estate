# PROJECT-1 REAL ESTATE

## PROJECT DESCRIPTION
Here in this project we are doing analysis on Real estate Price prediction using the Linear, Random Forest, xgboost and GBM model and choose the best model and prediction for Price. Also we are going to explain the entire project on the basis of Data Analysis procedure of Ask, Prepare, Process, Analyze, Share and Report using R-Programming.
## PROBLEM STATEMENT: ASK 
A real estate agency wanted to reduce the negotiation time and improve
closure for buyers and sellers of homes by ensuring that both sides were advised well on the potential
sale/purchase price of the home. To that end the agency wanted to predict a transaction price for all the
houses in its market which would be as close as possible to a price where the transaction would take place.
Data regarding all possible variables and qualities of over 10,000 houses sold in the past was collected and
analyzed.
## COLLECT AND IMPORTING THE DATASET: PREPARE
Once i have collected the data from stakeholder i have received the Dataset which has 7536 observations and 16 variables. 
In the initial step im importing the  Dataset to the R and started to work on the Dataset.

```getwd()
setwd("C:/edvancer/R programing/case study 1 dataset")  #Setting the working directory
hs_train=read.csv("housing_train (4).csv",stringsAsFactors = FALSE)  #Importing the Train dataset
```

## DATA CLEAINING PROCEDURE : PROCESS
After importing the Dataset we should understand the Dataset so that we can identify that the  NA values, Dummy creation, Unnecessary Variables, Converting variables to numeric / Factor/ Integer and other errors in the dataset.

```view(head(hs_train))``` #Train dataset

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/03fea638-1289-464e-a8a4-688287c2e568)


### Deal with NA values

After importing the dataset we need to understand the Missing values and NA's 

```colSums(is.na(hs_train))``` #We have get the data of NA values accumulated in Train data

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/6aa6e4ae-28c2-41ce-b515-5d7b982e2267)

We can also plot the Tables of NAs using the ```vis_dat``` function.

``` vis_dat(hs_train) # to plot the train dataset ```

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/f2d27be6-11d6-443c-aa2c-dd5964fe0108)


There are many missing values and we need to convert some Variables into Numerical values, And We need to create Dummies for certain Variables.

As per the Variable Analysis i have found some inputs and applied it on the dataset.

```
#Suburb :  drop the variable as this has no correlation.
#Address : drop the variable as this has no correlation.
#Rooms : club low frequency cateorories and then create dummies then conver to numeric.
#Type : create dummies then convert to numeric.
#Price :  This is the target variable, price of the property. Convert to numeric.
#Method : convert numeric:: convert to numeric club low frequency cateorories and then create dummies.
#SellerG : drop the variable as this has no correlation.
#Distance : converting it to numeric. 
#Postcode :  drop the variable as this has no correlation.
#Bedroom2 :  create dummies then convert to numeric.
#Bathroom : create dummies then convert to numeric.
#Car : create dummies then convert to numeric.
#Landsize : convert numeric :: convert to numeric.
#BuildingArea : convert to numeric.
#YearBuilt : drop the variable as this has no correlation.
#CouncilArea : create dummies then convert to numeric.
```

### Data Cleaning Process: 

Im doing some data cleansing procedure using recipe functions

Aim :
1. To remove the unwanted variables
2. Convert the Variables into Numeric/Charecter/Factor.
3. To create dummies
4. To handle the missing values followed by Mean, Median, mode etc.

```
#converting target var as numeric
hs_train$Price=as.numeric(hs_train$Price)
```

```
library(tidymodels) #here i imported the library called tidymodels
dp_pipe=recipe(Price ~ .,data=hs_train) %>% 
  update_role(Suburb,Address,SellerG,Postcode,YearBuilt,new_role = "drop_vars") %>% 
  update_role(Rooms,Landsize,
              Bedroom2,Car,BuildingArea,CouncilArea,
              Bathroom,new_role="to_numeric") %>% 
  update_role(Type, Method, CouncilArea,new_role="to_dummies") %>%  
  step_rm(has_role("drop_vars")) %>%
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_mutate_at(has_role("to_numeric"),fn=as.numeric) %>% 
  step_impute_median(all_numeric(),-all_outcomes())
```
Once the ```dp_pipe``` is created we need to ```prep``` and ```bake``` the dp_pipe in the next step

```
dp_pipe=prep(dp_pipe)
hs_train=bake(dp_pipe,new_data = hs_train)
```
After the recipe funtion there will be no Na or missing values and also there will additional variables were added due to the Create Dummy funtion. Lets check the dataset.

again im running the is.na querry to check the NA's

``` colSums(is.na(hs_train))#We have get the data of NA values accumulated in Test data ```

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/0f456b7c-d291-4cf1-81b3-65b33d403ef0)

``` vis_dat(hs_train) # to plot the train dataset ``` 

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/4c5f6a24-daaa-4922-9969-c27325f1b87a)

We can see there is no missing values, all the Variables are converted into  numerical values and Dummies are created.

### Splitting the Dataset
Split the `hs_train` dataset into two `trn` and `tst` to check the model performance.

```
set.seed(2) #using this function system cannot ablter any rows throughout the process.
s=sample(1:nrow(hs_train),0.8*nrow(hs_train)) #we are spliting the data into 80% and 20%
trn=hs_train[s,]
tst=hs_train[-s,]
```

One the Dataset has been splitted we can check the performance of dataset using Different Models.

### Implement and Check with the different models

We are imposing various models to check the performance of the Dataset.

#### Linear Model:

```
fit = lm(Price~.,data=trn)
```

Once we run the Model we need to check the model performance using Root Mean Squared Error(RMSE) and Mean Absolute Error (MAE)of Both `tst` and `trn` dataset.

#### Checking the RMSE and MAE of `tst` datset

```
test.pred=predict(fit,newdata = tst)

errors=tst$Price-test.pred

rmse = (errors)**2 %>% mean() %>% sqrt() #431630.9

mae=mean(abs(errors)) #277454.6
```
#### Checking the RMSE and MAE of `trn` datset

```
train.pred=predict(fit,newdata = trn)

errors=trn$Price-train.pred

rmse = (errors)**2 %>% mean() %>% sqrt() #417847.7

mae=mean(abs(errors)) #283665.2
```
We can observe that the Residual standard error: 418900 on 5998 degrees of freedom
And the Multiple R-squared  0.5997 $  Adjusted R-squared:  0.5977.
RMSE: 431630.9
MAE: 283665.2

#### Gradient Boosting Machine

```
gbm.fit=gbm(Price~.-Type_X__other__,
            data=trn,
            distribution = "gaussian", 
            n.trees = 500,
            shrinkage = 0.01,
            bag.fraction = 0.9,
            interaction.depth = 3)  

help("gbm")
```
#### Checking the RMSE and MAE of `tst` datset

```
test.predicted.gbm=predict.gbm(gbm.fit,newdata=tst)

errors=tst$Price-test.predicted.gbm

rmse = (errors)**2 %>% mean() %>% sqrt() #426854.8

mae=mean(abs(errors)) #271378
```
#### Checking the RMSE and MAE of `trn` datset

```
train.predicted.gbm=predict.gbm(gbm.fit,newdata=trn)

errors=trn$Price-train.predicted.gbm

rmse = (errors)**2 %>% mean() %>% sqrt() #408792.7

mae=mean(abs(errors)) #271378
```
`summary(gbm.fit)`

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/c14a2fe3-5bd1-4235-a482-a216b8533dde)

We can see the Rooms and u-type of the properties are having High Relative Influence

RMSE: 426854.8
MAE: 271378

#### XGBoost Model   
In this model we are  we are seperating the target variable and independent variable and creating x_train and y_train then we will implementing it on the model.

```
x_train=trn %>% select(-Price)
y_train=trn$Price
x_test=tst %>% select(-Price)
xgb.fit=xgboost(data=data.matrix(x_train),
                label = y_train,
                objective='reg:linear',
                verbose=1,
                nrounds = 10)
```

#### Checking the RMSE and MAE of `tst` datset
```
test.predicted.xgb=predict(xgb.fit,data.matrix(x_test))

errors=tst$Price-test.predicted.xgb

rmse = (errors)**2 %>% mean() %>% sqrt() #390284

mae=mean(abs(errors)) #231787.1
```

#### Checking the RMSE and MAE of `trn` datset
```
train.predicted.xgb=predict(xgb.fit,data.matrix(x_train))

errors=trn$Price-train.predicted.xgb

rmse = (errors)**2 %>% mean() %>% sqrt() #318306.2

mae=mean(abs(errors)) #201579.3

```
RMSE: 390284
But If we take the difference of GBM `tst` and `trn` it would be 20.3158% and there is unstability in the dataset.

### RANDOM FOREST

Let`s check the model performance on RandomForest

```
rf = randomForest(Price~.,data = trn, ntree = 100,
                  nodesize=400,
                  maxnodes=30,
                  do.trace=TRUE)
```
#### Checking the RMSE and MAE of `tst` datset
```
test.predicted.rf=predict(rf, newdata=tst)

errors=tst$Price-test.predicted.rf

rmse = (errors)**2 %>% mean() %>% sqrt() #435917.7

mae=mean(abs(errors)) #273009.8
```
#### Checking the RMSE and MAE of `trn` datset
```
train.predicted.rf=predict(rf, newdata=trn)

errors=trn$Price-train.predicted.rf

rmse = (errors)**2 %>% mean() %>% sqrt() #418170.9

mae=mean(abs(errors)) #267736.6
```
`VarImpPlot(rf)`

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/d813151a-229b-45e7-8ccd-135d719f8560)

From the above output we can see The Rooms,Type U, BuildingArea, Distance having high correlation with Price.

RMSE: 435917.7
MAE: 273009.8

once we are completed the model performance we can see Linear model, Random Forest and GBM models are doing good on this perticular dataset. So in order to bring down the RMSE and MAE  value we need to follow the CvTuning with KFold validation.

### KFold Cross Validation
Using this Tuning our model will behave stable in any given condition. Here in this Tuning it helps to create different combination of data and we are checking the stability of the dataset.

In the begininga stage we are stacking with  the funtion implementation.

```
library(CvTools)
mykfolds=function(nobs,nfold=5){
  
  t=cvFolds(nobs,K=nfold,type='random')
  
  folds=list()
  
  for(i in 1:nfold){
    
    test=t$subsets[t$which==i]
    train=t$subsets[t$which!=i]
    
    folds[[i]]=list('train'=train,'test'=test)
  }
  
  return(folds)
}

myfolds=mykfolds(nrow(trn),10) #here i applied the myfolds funtion on `trn` dataset
```
In the next stage we are creating the empty data frame with 3 columns for implimentation of 'trn' dataset.

### Stack layer 2  data  for train
```
bd_train_layer1=data.frame(rf_var=numeric(nrow(trn)),
                           xgb_var=numeric(nrow(trn)),
                           gbm_var=numeric(nrow(trn)))
```
Now we are implementing the XGboost, GBM and RandomForest to the KFold functionof Layer 1
```
for(i in 1:10){
  print(c(i))
  fold=myfolds[[i]]
  
  train_data=trn[fold$train,]
  test_data=trn[fold$test,]
  
  print('rf')
  
  rf.fit=randomForest(Price~.-Type_X__other__,data=train_data,
                      ntree=100,mtry=10)
  rf_score=predict(rf.fit,newdata=test_data)
  
  print('gbm')
  gbm.fit=gbm(Price~.-Type_X__other__,
              data=train_data,
              distribution = "gaussian",
              n.trees = 100,interaction.depth = 3)
  
  gbm_score=predict(gbm.fit,newdata=test_data,
                    n.trees=100)
  
  
 print('xgb')
  x_train=trn %>% select(-Price)
  y_train=trn$Price
  x_test=tst %>% select(-Price)
  xgb.fit=xgboost(data=data.matrix(x_train),
                  label = y_train,
                  objective='reg:linear',
                  verbose=1,
                  nrounds = 10)
  xgb_score=predict(xgb.fit,data.matrix(x_test))
  
  
  bd_train_layer1$rf_var[fold$test]=rf_score
  
  bd_train_layer1$gbm_var[fold$test]=gbm_score
  
  bd_train_layer1$xgb_var[fold$test]=xgb_score
}
```
Once run this the iteration will get started as per our input.
Similarly we should do the Layer 2 for 'tst' dataset

### Stack layer 2  data  for test
```
bd_test_layer2=data.frame(rf_var=numeric(nrow(tst)),
                          xgb_var=numeric(nrow(tst)),
                          gbm_var=numeric(nrow(tst)))
```
### now we are running the model on full train data (trn)

RandomForest:
```
full.rf=randomForest(Price~.-Type_X__other__,data=trn, 
                     ntree=100,mtry=10)
bd_test_layer2$rf_var=predict(full.rf,newdata=tst)
```
Gradient Boosting Machine:
```
full.gbm=gbm(Price~.-Type_X__other__,
             data=trn,
             distribution = "gaussian",
             n.trees = 100,interaction.depth = 3)
bd_test_layer2$gbm_var=predict(full.gbm,newdata=tst,
                               n.trees=100)
```
XGBoost:
```
x_train=trn %>% select(-Price)
y_train=trn$Price
x_test=tst %>% select(-Price)
xgb.fit=xgboost(data=data.matrix(x_train),
                label = y_train,
                objective='reg:linear',
                verbose=1,
                nrounds = 10)
bd_test_layer2$xgb_var=predict(xgb.fit,data.matrix(x_test))
```
Linear Model:
Now we are running Linear model on top of it to get the final output.

```
bd_train_layer1$Price=trn$Price #we are impleminting Layer 1 data set to `trn$price`
bd_test_layer2$Price=tst$Price #we are impleminting Layer 1 data set to `tst$price`

lin.model=lm(Price~.,data=bd_train_layer1) #running the Linear models on the layer 1 and implementing on layer 2
```
### Checking the RMSE and MAE of `tst` datset of layer2
```
test.predicted.l2=predict(lin.model,newdata = bd_test_layer2)

errors=bd_test_layer2$Price-test.predicted.l2

rmse = (errors)**2 %>% mean() %>% sqrt() #356692.6

mae=mean(abs(errors)) #210985
```
### Checking the RMSE and MAE of `trn` datset of layer1 
```
train.predicted.l1=predict(lin.model,newdata = bd_train_layer1)

errors=bd_train_layer1$Price-train.predicted.l1

rmse = (errors)**2 %>% mean() %>% sqrt() #346279.4

mae=mean(abs(errors)) #211363
```
`summary(lin.model)`

We can observe that the Residual standard error: 346400 on 6024 degrees of freedom
Multiple R-squared:  0.7251,	Adjusted R-squared:  0.7249 
RMSE: 356692.6
MAE: 210985
We can observe that the after KFold cross validation our test and train data's are performed very well on these models and there is observable results in the RMSE and R-Square values. 

## Data observation : Analyze and Share the data
Standard error: we can cledarly see that there is a considerable difference in the standard error as in the initial stage the SE was 418900 on 5998 degrees of freedom and after the Iternation this bring down to the value of 346400 on 6024 degrees of freedom which is a quite good result.
R squared error: Similarly we can see the R-Squared error also giving a decent result. Before the iteration the R-Squared error was 0.5997 and once the proper iteration is done the value bring up to the 0.7249 which is decent value.

## Solution statement: Act
Created a predictive model using linear regression and implemented CvTuning with the help of
Random forest and GBM and XGBoost to arrive at a potential transaction price for all future transactions. As per the Data Anlaysis Over the next 6 months, negotiation time was brought down from 27 days on average to 8 days which is almost a save the 3 weeks for the real estate agents. and the percentage of the Deal closure went up by 19%.  























































































































