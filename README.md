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
Once i have collected the data from stakeholder i have received two datasets, in that one is Train which has Target and Dependent variables and another Test dataset has only Dependent variables and we should predict our Train data on this. 
In the initial step im importing the Train and Test Dataset to the R.

```getwd()
setwd("C:/edvancer/R programing/case study 1 dataset")  #Setting the working directory
hs_train=read.csv("housing_train (4).csv",stringsAsFactors = FALSE)  #Importing the Train dataset
hs_test=read.csv("housing_test (2).csv",stringsAsFactors = FALSE) #Importing the Test dataset
```

## DATA CLEAINING PROCEDURE : PROCESS
After importing the Dataset we should understand the Dataset so that we can identify that the  NA values, Dummy creation, Unnecessary Variables, Converting variables to numeric / Factor/ Integer and other errors in the dataset.

```view(head(hs_train))``` #Train dataset

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/03fea638-1289-464e-a8a4-688287c2e568)

```view(head(hs_test))``` #Test dataset

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/591ab987-7a99-4a55-97d7-67e8acd9f6bd)

```colSums(is.na(hs_test))``` #We have get the data of NA values accumulated in Test data

![image](https://github.com/swasthik62/Project1-Real-Estate/assets/125183564/6aa6e4ae-28c2-41ce-b515-5d7b982e2267)

As per the variable analysis i have found some inputs and applied it on the dataset.

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
Im doing some data cleansing procedure using recipe functions

```
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
hs_test=bake(dp_pipe,new_data = hs_test)
```





































































































































