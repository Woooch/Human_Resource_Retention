library(tidyverse)
library(caret)
library(ggplot2)
library(corrplot)

### 1. Inserting Data
df = read.csv("C:/Users/Willy/Downloads/HRDataset_v14.csv", header= T, stringsAsFactors = TRUE)
names(df)
attach(df)
head(df)
dim(df)
summary(df)
glimpse(df)

#_______________________________________________________________________________#

### 2. Data Cleaning / Process
sum(is.na(df))
df = na.omit(df)
sum(duplicated(df))
sum(is.na(df)) #Checking

#Scaling things so it makes sense
scaled.Salary = scale(df$Salary)
scaled.EngagementSurvey = scale(df$EngagementSurvey)
scaled.EmpSatisfaction = scale(df$EmpSatisfaction)
scaled.SpecialProjectsCount = scale(df$SpecialProjectsCount)
scaled.DaysLateLast30 = scale(df$DaysLateLast30)
scaled.Absences = scale(df$Absences)

#Turning varaibles that represent a interger into a factor because they are categorical
columns_to_convert <- c("EmpID", "MarriedID", "MaritalStatusID", "GenderID", "EmpStatusID", "DeptID", 
                        "PerfScoreID", "FromDiversityJobFairID","Termd", "PositionID", "ManagerID", "Zip")
df[, columns_to_convert] <- lapply(df[, columns_to_convert], factor)

#Dropping columns
cols_to_drop = c("Zip","DateofHire","DateofTermination","Employee_Name","ManagerName","LastPerformanceReview_Date","RecruiterSource")
df = subset(df, select = -cols_to_drop)
df = df %>% select(-cols_to_drop)
#_______________________________________________________________________________#

### 3. Exploratory Data Analysis
#Employee Demographic
op=par(mfrow=c(2,2))
plot(Sex,col = "lightblue", xlab = "Gender", ylab= "Count") #Sex
plot(RaceDesc,col = "lightblue",xlab = "Ethnicity", ylab= 'Count', ) #Ethnicity
plot(State, col = "lightblue", xlab = "States", ylab= "Count", ) #State demography
plot(MaritalDesc,col = "lightblue", xlab= "Status", ylab='Count') #Marital status

#Employee termination reasons
op=par(mfrow=c(1,2))
filtered_data <- df[df$TermReason != "N/A-StillEmployed",]
term_reason_counts <- table(filtered_data$TermReason)
barplot(term_reason_counts, col = "lightblue",  #
        xlab= "Termination Reason excluding those still employed", 
        ylab='Count')
emp_stat = table(df$EmploymentStatus)
barplot(emp_stat, col='lightblue', xlab ='Employment Status', ylab='Count')

#Role and Departments
plot(df$Position, col = "lightblue",xlab= "Status", ylab='Count' )
plot(Department, col = "lightblue")    
plot(df$Salary, col = 'blue')

#Employee/Performance analysis
op=par(mfrow=c(1,3))
plot(PerformanceScore, col = "lightblue")
plot(EmpSatisfaction, col= "lightblue")
plot(EngagementSurvey, col = "lightblue")

op=par(mfrow=c(1,2))
plot(EmpSatisfaction,EngagementSurvey, col = 'lightblue')
plot(Salary, PerformanceScore, col='lightblue', xlab = "Salary", 
     ylab= 'PerformanceScore', main = 'Performance Score vs Salary')

plot(Salary, EmpSatisfaction, col='lightblue', xlab = "Salary", 
     ylab= 'EmpSatisfaction', main = 'EmpSatisfaction vs Salary')

op=par(mfrow=c(1,1))
#State demographic
state_counts = table(df$State)
state_counts
Position_counts = table(df$Position)
Position_counts


#Correlation Heatmap (Not very useful, hard to read)
correlation_matrix <- cor(df[, sapply(df, is.numeric)], 
                          use = "pairwise.complete.obs")
corrplot(correlation_matrix,
         method = "color", 
         type = "lower", 
         order = "hclust", 
         addrect = 2)

table(df$MaritalStatusID,df$MaritalDesc)


# 5. Model Building

# Data Split 80-20
set.seed(1234)
train.index = sample(nrow(df),nrow(df)*0.8)
df.train = df[train.index,]
df.test  = df[-train.index,]

#_______________________________________________________________________________#

# 5.1 Logistic regression
#Goal: Build a model to predict TERMD from the other variables

LogReg = glm(Termd ~  SpecialProjectsCount + Salary + EmpSatisfaction + EngagementSurvey + PerformanceScore + DaysLateLast30,
             data =df.train,
             family = 'binomial')
summary(LogReg)

Prediction_LogReg = predict(LogReg, df.test, type = 'response')
threshold = 0.5
Predicted_class = ifelse(Prediction_LogReg > threshold, 1, 0)

actual_class = df.test$Termd
confusionMatrix(factor(Predicted_class),actual_class)
varImp(LogReg)

#_______________________________________________________________________________#

# 5.2 Logistic regression Caret          (This was the one writing in the report)
# Define the logistic regression model formula (replace with your actual formula)

control = trainControl(method = "repeatedcv", 
                       number = 10, 
                       repeats = 3)  

LogReg2_formula = formula(Termd ~  SpecialProjectsCount + Salary + EmpSatisfaction + EngagementSurvey + PerformanceScore + DaysLateLast30)

LogReg2 = train(LogReg2_formula, 
                data = df.train, 
                method = "glm", 
                family = "binomial", 
                trControl = control)
summary(LogReg2)
predictions_2 <- predict(LogReg2, df.test, type="raw")
confusionMatrix(predictions_2,actual_class)

varImp(LogReg2)

#_____________________________________________________________________________#
# 5.4 Random Forest Model                           (Used in report)

control = trainControl(method = "cv", 
                       number = 5)  
grid_rf = expand.grid(mtry = seq(1,5))

rf_formula = formula(Termd~ SpecialProjectsCount + Salary + EmpSatisfaction + EngagementSurvey + PerformanceScore + DaysLateLast30)
randomforest = train(rf_formula,
                     data = df.train,
                     trControl = control,
                     method = "rf",
                     tuneGrid = grid_rf)
summary(randomforest)
plot(randomforest)

prediction_4 = predict(randomforest, df.test, type="raw")
confusionMatrix(prediction_4,actual_class)

varImp(randomforest)
library(tree)
library(rpart)
prune.model=prune.rpart(randomforest,cp=5)

plot(prune.model)
text(prune.model,pretty=0)
#_____________________________________________________________________________#

#5.5 Random Forest with different formula including addition predictors
rf_formula_2 = formula(Termd~ SpecialProjectsCount + EmpSatisfaction + EngagementSurvey + PerfScoreID + Salary + Absences + PositionID)
randomforest_2 = train(rf_formula_2,
                     data = df.train,
                     trControl = control,
                     method = "rf",
                     tuneGrid = grid_rf)
randomforest_2
summary(randomforest_2)
plot(randomforest_2)
#_____________________________________________________________________________#

#5.6 Randomm Forest visualization   (Used in powerpoint and report)
library(tree)
set.seed(1234)

tree.model = tree(Termd~ SpecialProjectsCount + Salary + EmpSatisfaction + EngagementSurvey + PerformanceScore + DaysLateLast30 ,
                  df.train)


cv_model = cv.tree(tree.model, K = 10, FUN =  prune.misclass)
cv_model

prediction_5 = predict(cv_model, df.test, type = 'class')
confusionMatrix(prediction_5,actual_class, dnn = c("Active","Inactive"))


prune.model=prune.tree(tree.model,best=10)
plot(prune.model)
text(prune.model,pretty=0)

table(df$TermReason)
#_____________________________________________________________________________#

# 5.7 Gradient Boosting
library(gbm)
gb_formula = formula(Termd~ SpecialProjectsCount + Salary + EmpSatisfaction + EngagementSurvey + PerformanceScore + DaysLateLast30)
gradientboosting = train(gb_formula,
                         data = df.train,
                         trControl = control,
                         method = "gbm",
                         tuneLength = 5)


summary(gradientboosting)

prediction_7 = predict(gradientboosting, df.test, type = "raw")
confusionMatrix(prediction_7, actual_class)

vImpGBM = varImp(gradientboosting)


table(df$DateofTermination)

#_____________________________________________________________________________#
# 5.1 Logistic regression
#Goal: Build a model to predict TERMD from the other variables

LogReg = glm(Termd ~. ,
             data =df.train,
             family = 'binomial')
summary(LogReg)

Prediction_LogReg = predict(LogReg, df.test, type = 'response')
threshold = 0.5
Predicted_class = ifelse(Prediction_LogReg > threshold, 1, 0)

actual_class = df.test$Termd
confusionMatrix(factor(Predicted_class),actual_class)
varImp(LogReg)

table(df$DateofTermination)
