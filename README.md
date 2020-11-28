---
title: "Random Forest Lab"
author: "Max St John, Helena Lindsey, Allen Baiju"
date: "11/16/2020"
output: html_document
editor_options: 
  chunk_output_type: console
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r, include=FALSE}
library(tidyverse)
library(randomForest)
library(rio)
library(knitr)
library(plotly)
library(caret)
library(pROC)
library(ROCR)
```


Your world renowned work in cancer research, sports recruiting, advertising, and environmental research has lead the government to reach out an ask for help in better understanding the US populous.  Your goal is to build a classifier that can predict the income levels in order to create more effective policy.  

The dataset below includes Census data on 32,000+ individuals with a variety of variables and a target variable for above or below 50k in salary. 

Your goal is to build a Random Forest Classifier to be able to predict income levels above or below 50k. 




```{r, include=FALSE}
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
lables <- c("age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income")
census <- read_csv(url, col_names = lables, col_types = cols(
  age = col_double(),
  workclass = col_factor(),
  fnlwgt = col_double(),
  education = col_factor(),
  education_num = col_double(),
  marital_status = col_factor(),
  occupation = col_factor(),
  relationship = col_factor(),
  race = col_factor(),
  sex = col_factor(),
  capital_gain = col_double(),
  capital_loss = col_double(),
  hours_per_week = col_double(),
  native_country = col_factor(),
  income = col_factor()
))
```


```{r, include=FALSE}
#Refactor the target variable to set the above 50k to 1 and below to 0. 
census$income = ifelse(census$income == ">50K", 1, 0)
census$income <- as.factor(census$income)
```


```{r, include=FALSE}
#Ensure that the variables are correctly classified ()
census$native_country <- ifelse(census$native_country == "?", NA, census$native_country)
census <- na.omit(census)
census$native_country <- as.factor(census$native_country)
```

Calculate the base rate
```{r}
length <- length(filter(census, income == "1")$income)
baserate <- length/length(census$income)
baserate
```


```{r, include=FALSE}
#Create test and training sets 
sample_rows = 1:nrow(census)
set.seed(1984) 
test_rows = sample(sample_rows,
                   dim(census)[1]*.10, #start with 10% of our dataset, could do 20%
                   # but random forest does require more training data because of the 
                   # sampling so 90% might be a better approach with this small of a dataset
                   replace = FALSE)# We don't want duplicate samples
# Partition the data between training and test sets using the row numbers the
# sample() function selected.
census_train = census[-test_rows,]
census_test = census[test_rows,]
```

# Models{.tabset}
## Initial RF Model
### Initial mtry
Calculate the initial mtry level 
```{r}
mtry_initial <- round(sqrt(length(census)))
mtry_initial
```
### Running initial model
Run the initial RF model with 500 trees 
```{r, include=FALSE}
census_RF = randomForest(income~.,
                            census_train,     
                            ntree = 500,        
                            mtry = mtry_initial,            
                            replace = TRUE,      
                            sampsize = 100,      
                            nodesize = 5,        
                            importance = TRUE,   
                            proximity = FALSE,    
                            norm.votes = TRUE,   
                            do.trace = TRUE,     
                            keep.forest = TRUE,  
                            keep.inbag = TRUE)
```

```{r}
census_RF
```

### Initial Model Analysis
Review the percentage of trees that voted for each data point to be in each class.
```{r}
census_votes <- as.data.frame(census_RF$votes)
kable(head(census_votes))
```

Review the "predicted" argument contains a vector of predictions for each 
data point.
```{r}
head(as.data.frame(census_RF$votes))
head(as.data.frame(census_RF$predicted))
```

Determine the most important variable for your model 
```{r}
census_importance_DF <- as.data.frame(census_RF$importance)
kable(census_importance_DF[-c(1:2)])
```

Occupation is the most important variable with the highest mean decrease in Gini index, 4.8070253. 


Visualize the model using plotly with lines tracking the overall oob rate and for the two classes of the target variable. What patterns do you notice?
```{r}
census_RF_error = data.frame(1:nrow(census_RF$err.rate),
                                census_RF$err.rate)
colnames(census_RF_error) = c("Number of Trees", "Out of the Box",
                                 "<=50K", ">50K")
census_RF_error$Diff <- census_RF_error$`>50K`-census_RF_error$`<=50K`
library(plotly)
fig <- plot_ly(x=census_RF_error$`Number of Trees`, y=census_RF_error$Diff,name="Diff", type = 'scatter', mode = 'lines')
fig <- fig %>% add_trace(y=census_RF_error$`Out of the Box`, name="OOB_Er")
fig <- fig %>% add_trace(y=census_RF_error$`<=50K`, name="<=50K")
fig <- fig %>% add_trace(y=census_RF_error$`>50K`, name=">50K")
fig
```
All graphs have a high variance with a low number of trees, but they begin to level out as the number of trees grow. In addition, the error for the >50k class is much higher than the <=50k class. 

Pull the confusion matrix out of the model and discuss the results 
```{r}
census_RF$confusion
census_RF_acc = sum(census_RF$confusion[row(census_RF$confusion) == 
                                                col(census_RF$confusion)]) / 
  sum(census_RF$confusion)
census_RF_acc
```

The accuracy of the model is 84.33%.


Determine which tree created by the model has the lowest out of bag error

```{r}
min_OOB <- min(census_RF_error$`Out of the Box`)
number_trees <- which(census_RF_error == min_OOB, arr.ind=TRUE)[,1]
number_trees
```

Using the error.rate matrix select the number of trees that appears to minimize your error.

```{r}
min_OOB_matrix <- min(census_RF$err.rate[,1])
number_trees_matrix <- which(census_RF$err.rate == min_OOB, arr.ind=TRUE)[,1]
number_trees_matrix
```

## Min OOB Model
### Running Min OOB Model
Using the visualizations, oob error and errors for the poss and negative class, select a new number of trees and rerun the model
```{r, include=FALSE}
census_RF_2 = randomForest(income~.,
                            census_train,     
                            ntree = 93,        
                            mtry = mtry_initial,            
                            replace = TRUE,      
                            sampsize = 300,      
                            nodesize = 5,        
                            importance = TRUE,   
                            proximity = FALSE,    
                            norm.votes = TRUE,   
                            do.trace = TRUE,     
                            keep.forest = TRUE,  
                            keep.inbag = TRUE)
```

```{r}
census_RF_2
```

### Comparing two models
Compare the new model to the original and discuss the differences
```{r}
census_RF$confusion
census_RF_2$confusion
census_RF_2_acc = sum(census_RF_2$confusion[row(census_RF_2$confusion) == 
                                                col(census_RF_2$confusion)]) / 
  sum(census_RF_2$confusion)
census_RF_2_acc
```

The accuracy for the second model is 85.04%, and is 0.71% higher than the first model with a rate of 84.33%.

### Min OOB Model Analysis
Use the better performing model to predict with your training set
```{r}
census_predict = predict(census_RF_2,      #<- a randomForest model
                            census_test,      #<- the test data set to use
                            type = "response",   #<- what results to produce, see the help menu for the options
                            predict.all = TRUE,  #<- should the predictions of all trees be kept?
                            proximity = TRUE)
```

Use the confusion matrix function to assess the model quality 
```{r}
census_test_pred = data.frame(census_test, 
                                 Prediction = census_predict$predicted$aggregate)
confusionMatrix(census_test_pred$Prediction,census_test_pred$income,positive = "1", 
                dnn=c("Prediction", "Actual"), mode = "everything")
```

Create a variable importance plot and discuss the results
```{r}
census2_importance_DF <- as.data.frame(census_RF_2$importance)
kable(census2_importance_DF[-c(1:2)])
varImpPlot(census_RF_2,     
           sort = TRUE,        
           n.var = 10,        
           main = "Important Factors for Identifying Income Range",
           #cex = 2,           #<- size of characters or symbols
           bg = "white",       
           color = "blue",     
           lcolor = "orange")
```

Occupation has the highest mean decrease in Gini index and the second highest mean decrease in accuracy. This highlights how important the occupation variable is. Relationship is also a very important variable, with the second-highest mean decrease in Gini index and the third-highest mean decrease in accuracy. 


Use the tuneRf function to select a optimal number of variables to use during the tree building process, if necessary rebuild your model and compare the results the previous model. 
```{r}
set.seed(2)
census_RF_mtry = tuneRF(data.frame(census_train[ ,1:14]),  
                           as.factor(census_train$income),     
                           mtryStart = 5,                        
                           ntreeTry = 100,                       
                           stepFactor = 2,                      
                           improve = 0.05,                       
                           trace = TRUE,                         
                           plot = TRUE,                          
                           doBest = FALSE)
```

Create a graphic that shows the size of the trees being grown for your final model.
```{r}
hist(treesize(census_RF, terminal = TRUE), main="Tree Size")
```

Evaluate the final model using the performance/prediction functions in the ROCR package. 
```{r}
census_RF_prediction = as.data.frame(as.numeric(as.character(census_RF_2$votes[,2])))
census_train_actual = data.frame(census_train[,15])
census_prediction_comparison = prediction(census_RF_prediction, census_train_actual)
census_pred_performance = performance(census_prediction_comparison, 
                                         measure = "tpr",    #<- performance measure to use for the evaluation
                                         x.measure = "fpr")
census_rates = data.frame(fp = census_prediction_comparison@fp,  #<- false positive classification.
                             tp = census_prediction_comparison@tp,  #<- true positive classification.
                             tn = census_prediction_comparison@tn,  #<- true negative classification.
                             fn = census_prediction_comparison@fn) #<- false negative classification.
colnames(census_rates) = c("fp", "tp", "tn", "fn")
tpr = census_rates$tp / (census_rates$tp + census_rates$fn)
fpr = census_rates$fp / (census_rates$fp + census_rates$tn)
census_rates_comparison = data.frame(census_pred_performance@x.values,
                                        census_pred_performance@y.values,
                                        fpr,
                                        tpr)
colnames(census_rates_comparison) = c("x.values","y.values","fpr","tpr") #<- rename columns accordingly.
census_auc_RF = performance(census_prediction_comparison, 
                               "auc")@y.values[[1]]
plot(census_pred_performance, 
     col = "red", 
     lwd = 3, 
     main = "ROC curve")+
  grid(col = "black")+
  abline(a = 0, 
       b = 1,
       lwd = 2,
       lty = 2,
       col = "gray")+
  text(x = 0.5, 
     y = 0.5, 
     labels = paste0("AUC = ", 
                     round(census_auc_RF,
                           2)))
```

# Summary
Our final model was the model we made using the number of trees that minimized our OOB error. The model had an accuracy of .8621, much higher than the no information rate of .7698. Despite this high accuracy, we had a moderate kappa value of .5745. To get some more insights, we will look at our sensitivity and specificity. Our model struggled to predict the positive class, with a sensitivity of only .5788. On the other hand, we had a specificity of .9468. These values show how the accuracy of our model was likely inflated. The dataset was not balanced, with 24,283 individuals making less than $50k and only 7695 individuals making more than $50k. This means that despite struggling to predict the positive class, the success in predicting the negative class outweighs these struggles, inflating our accuracy. The model had an AUC of .9, but this high number is likely due to the issues we just highlighted. Looking at the Gini index and accuracy for each variable, occupation was clearly the most important, with education, age, and marital status also proving to be important.


