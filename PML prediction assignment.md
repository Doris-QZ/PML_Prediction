# Activity Prediction of Weight Lifting Exercises  

Doris Chen  
2023-02-16  

### Prepare

**Loading the data**

```{r, message=FALSE}
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url_train, destfile="/Users/Doris/Desktop/PML_Prediction/pml-training.csv", method="curl")
pml_training <- read.csv("pml-training.csv")

url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url_test, destfile="/Users/Doris/Desktop/PML_Prediction/pml-testing.csv", method="curl")
pml_testing <- read.csv("pml-testing.csv")
```
  
**Loading packages**  
  
```{r, message=FALSE}
library(caret)
library(ranger)
```
  
### Process  
  
Check the dimension and the column names of the data.  

```{r, results='hide'}
dim(pml_training)   
names(pml_training)
head(pml_training) # Results are hidden.
```
  
There are 19622 observations of 160 variables. Some columns with "NA" values and some with blank values.  
  
We convert blank values to "NA", then check the total missing values in the data.    
  
```{r, results='hide'}
pml_training[pml_training==""] <- NA
colSums(is.na(pml_training))   
sum(colSums(is.na(pml_training))==19216) # Results are hidden.
```
  
There are 100 columns containing "NA", each are 19216. We remove these columns together with column 1 to 7, which are not data from the accelerometers.  
  
```{r}
training <- pml_training[colSums(is.na(pml_training))==0]
training <- training[,-c(1:7)]
```
    
We perform the same process on test set.   
  
```{r}
pml_testing[pml_testing==""] <- NA
testing <- pml_testing[colSums(is.na(pml_testing))==0]
testing <- testing[,-c(1:7)]
```  
  
### Fit the model  
  
**1. Preprocess the data**  

Check the correlation between variables.  
  
```{r}
corr <- cor(training[,-53])
heatmap(corr)
```
  
The heatmap shows that there are some highly correlated variables. We'll check how many of them have the correlation coefficient greater than 0.8.

```{r}
unique(corr[abs(corr)>0.8 & abs(corr)!=1])
```
  
19 pairs of variables have correlation greater than 0.8. So we preprocess the data with principal components analysis.    
  
```{r}
set.seed(1)
prep <- preProcess(training[,-53], method="pca", thresh = 0.99)
trainPC <- predict(prep, training[,-53])
trainPC$classe <- as.factor(training$classe)
```
  
Preprocess the test set.  
  
```{r}
testPC <- predict(prep, testing[,-53])
testPC$problem_id <- testing$problem_id
```
  
**2. Cross validation setting**        
  
We use k-fold for cross validation, set k=3.  
  
```{r}
train_control<- trainControl(method="cv", number=3, classProbs=TRUE)
```
  
**3. Train the model**     
  
Our goal is to predict the type of activities, so we use random forest to train the model.
  
```{r, cache=TRUE, results='hide'}
fit <- train(classe ~., data=trainPC, method="ranger", trControl=train_control) # Results are hidden
```
   
**4. Check the prediction accuracy**
  
```{r}
confusionMatrix(predict(fit, trainPC), trainPC$classe)
```
  
The accuracy is 100%.  

**5. Estimate out of sample error**  
  
```{r}
fit$finalModel
```
  
The OOB prediction error is 0.044. So the estimated out of sample error is 4.4%  
  
### Predict on test set  
  
```{r}
predict(fit, testPC[,-37])
```
  





