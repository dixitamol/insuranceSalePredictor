
# ########################################################
# 			-- Loading libraries --
# ########################################################

# clear variables
closeAllConnections()
rm(list=ls())
cat("\014") 

library(RPostgreSQL)
library(knitr)
library(rmarkdown)
library(ggplot2)
library(ggplot2)
library(plyr)
library(dummies)
library(ROCR)
library(crossval)
# install.packages("xlsx")
library(xlsx)

# load data
read_data_DB = 0 
if (read_data_DB == 1) { # read data from DB
  # First execute file ReadDataFromDB.R
  source("C:\\Users\\~\\Caravan Insurance - R\\CODE\\ReadDataFromDB.R")
  df_DBtrain <- fetch( dbSendQuery(con, 
                                 "SELECT * FROM tt_caravan_v1"), n=Inf)
  
  df_DBtrain <- Rename_DF_caravan_toUpper ( df_DBtrain)
  df_DBtrain <- rename(df_DBtrain,c(CARAVAN="Purchase"))
  
  caravan_df_maindataset <- df_DBtrain
  caravan_df_train <- df_DBtrain
  df_DBtest <- fetch( dbSendQuery(con, 
                                 "SELECT * FROM tt_test_caravan"), n=Inf)
  df_DBtest <- Rename_DF_caravan_toUpper ( df_DBtest)
  df_DBtest <- rename(df_DBtest,c(CARAVAN="Purchase"))
  caravan_df_test <- df_DBtest

} else { # load data from file
	caravan_df_maindataset <- read.csv('C:\\Users\\~\\Desktop\\Caravan Insurance - R\\DATA\\TrainingData_V2.csv')
	caravan_df_train <- read.csv('C:\\Users\\~\\Desktop\\Caravan Insurance - R\\DATA\\TrainingData_V2.csv')
	caravan_df_test <- read.csv('C:\\Users\\~\\Desktop\\Caravan Insurance - R\\DATA\\TestData_V2 (comma).csv')
	caravan_df_maindataset <- rename(caravan_df_maindataset,c(Number_of_mobile_home_policies="Purchase"))
	caravan_df_train <- rename(caravan_df_train,c(Number_of_mobile_home_policies="Purchase"))
	caravan_df_test <- rename(caravan_df_test,c(Number_of_mobile_home_policies="Purchase"))
}

select_features = 1
if (select_features == 1) { # Select features for logistic regression

  keeps_Lau <- c("fire_policies", "private_third_party_insurance", "car_policies", "Customer_Subtype",
                           "Average_income", "Purchasing_power", "High_level_education", 
                           "Income_._30", "Social_class_A", "Number_of_boat_policies", "Lower_level_education", 
                           "boat_policies", "No_car", "Married", "social_security_insurance_policies", "Purchase")
  
  keeps <- c("car_policies","Purchasing_power","Married","Lower_level_education","fire_policies","Number_of_boat_policies", "Purchase") 
  
  caravan_df_maindataset <- caravan_df_maindataset[keeps]
	caravan_df_train <- caravan_df_train[keeps]
	caravan_df_test <- caravan_df_test[keeps]
	} 


#Check Shape of Data and List Out All Columns
str(caravan_df_maindataset)
#Sample of the Data
head(caravan_df_maindataset)

# Ensure there are no missing values.
paste0("Missing values: ", sum(is.na(caravan_df_maindataset)))

# Finding which features are numeric.
numeric_caravan <- which(sapply(caravan_df_maindataset,is.numeric))
str(caravan_df_maindataset[,numeric_caravan])

# All features are numeric

caravan_df_maindataset$Purchase <- factor(caravan_df_maindataset$Purchase, labels=c(0,1))

#Let's Check How Unbalanced Our Data Is By Counting The Number of 1s and 0s In The Feature Purchase

Non_Caravan_Insurance  <- sum(caravan_df_maindataset$Purchase == 0)
Caravan_Insurance_Holder <- sum(caravan_df_maindataset$Purchase == 1)

dat <- data.frame(
  Caravan_Insurance_Holders = factor(c("Non_Caravan_Insurance","Caravan_Insurance_Holder"), levels=c("Non_Caravan_Insurance","Caravan_Insurance_Holder")),
  Count = c( Non_Caravan_Insurance , Caravan_Insurance_Holder)
)

ggplot(data=dat, aes(x=Caravan_Insurance_Holders, y=Count, fill=Caravan_Insurance_Holders)) +
  geom_bar(colour="black", stat="identity")
  
# [7]
Frequency.Purchase <- data.frame(Purchase = levels(caravan_df_maindataset$Purchase), Count = as.numeric(table(caravan_df_maindataset$Purchase)))
Frequency.Purchase

# characteristics of the observations that actually bought the 'mobile home insurance'. 
# In [313]:
TrainDataset <-caravan_df_maindataset

car_policies <- sum(TrainDataset$Purchase == 1 & TrainDataset$car_policies  != 0)
Purchasing_power <- sum(TrainDataset$Purchase == 1 & TrainDataset$Purchasing_power  != 0)
Married <- sum(TrainDataset$Purchase == 1 & TrainDataset$Married  != 0)
fire_policies <- sum(TrainDataset$Purchase == 1 & TrainDataset$fire_policies  != 0)
Lower_level_education <- sum(TrainDataset$Purchase == 1 & TrainDataset$Lower_level_education  != 0)

dat <- data.frame(
  Selected_Features = factor(c("car_policies" , "Purchasing_power " , "Married" , "fire_policies"  , "Lower_level_education" ), levels=c("car_policies" , "Purchasing_power " , "Married" , "fire_policies"  , "Lower_level_education")),
  Count = c( car_policies  ,  Purchasing_power  , Married , fire_policies  , Lower_level_education ))

ggplot(data=dat, aes(x=Selected_Features, y=Count, fill=Selected_Features)) +
  geom_bar(colour="black", stat="identity")

##########################################################  
# ########################################################
# 			-- LOGISTIC REGRESSION --
# ########################################################
##########################################################

# 3. Classification using Logistic Regression - Unbalanced Data, Undersampled Data, Oversampled Data - Including Business Cost For Each Model
# Logistic Using Unbalanced Data
# In [209]:

str(caravan_df_test)
paste0("Missing values: ", sum(is.na(caravan_df_test)))

#LR Pre-Processing
caravan_df_trainLR <- caravan_df_train
caravan_df_testLR <- caravan_df_test
caravan_df_trainLR$Purchase <- factor(caravan_df_trainLR$Purchase, labels = c(0,1))

caravan_df_trainLR <- dummy.data.frame(caravan_df_trainLR, sep = ".", names = c("Customer_main_type","Customer_Subtype"))
caravan_df_testLR <- dummy.data.frame(caravan_df_testLR, sep = ".", names = c("Customer_main_type","Customer_Subtype"))

# caravan_df_trainLR <- dummy.data.frame(caravan_df_trainLR, sep = ".", names = c("Purchasing_power"))
# caravan_df_testLR <- dummy.data.frame(caravan_df_testLR, sep = ".", names = c("Purchasing_power"))

# FINAL SET !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

rm(caravan_df_maindataset)
rm(caravan_df_test)
rm(caravan_df_train)

Select_subset =0

if (Select_subset == 1){
  
  caravan_df_trainLR <- subset(caravan_df_trainLR, select = c(car_policies ,Average_income,Purchasing_power ,private_third_party_insurance ,
                                                              fire_policies ,Number_of_boat_policies ,social_security_insurance_policies ,Lower_level_education ,
                                                              High_level_education  ,No_car ,Customer_Subtype.38 ,Customer_Subtype.37 ,
                                                              Customer_Subtype.3 ,Customer_Subtype.39 ,Customer_Subtype.36 ,Customer_Subtype.12 ,Customer_Subtype.8 ,
                                                              Customer_Subtype.20 ,Customer_Subtype.33 ,
                                                              Purchase))
  caravan_df_testLR <- subset(caravan_df_testLR, select = c(car_policies ,Average_income, Purchasing_power ,private_third_party_insurance ,
                                                            fire_policies ,Number_of_boat_policies ,social_security_insurance_policies ,Lower_level_education ,
                                                            High_level_education  ,No_car ,Customer_Subtype.38 ,Customer_Subtype.37 ,
                                                            Customer_Subtype.3 ,Customer_Subtype.39 ,Customer_Subtype.36 ,Customer_Subtype.12 ,Customer_Subtype.8 ,
                                                            Customer_Subtype.20 ,Customer_Subtype.33 ,
                                                            Purchase))
}

# Lower level education greater and smaller than 5 ... ???

#Classify Using Logistic Regression
logisticTrainingFit <- glm(Purchase ~ ., family = "binomial", data = caravan_df_trainLR)

  
#In [217]: Predict Class And Display Confusion Matrix
#Predict Class And Display Confusion Matrix
LOutPredicted <- predict(logisticTrainingFit, caravan_df_testLR, type = "response")
LOutPredictedClass <- ifelse(LOutPredicted>0.5, 1, 0)
LOutActual <- caravan_df_testLR$Purchase
LConfusionOutPredicted <- table(LOutActual, LOutPredictedClass)
rownames(LConfusionOutPredicted) <- c("0","1")
colnames(LConfusionOutPredicted) <- c("0","1")
LConfusionOutPredicted
for (i in 1:4000){
  probabilityDF$LR[i] <- LOutPredicted[i]
}
  
# Plot ROC and AUC for LR
probs <- LOutPredicted
library(ROCR)

LRPred <- prediction(probs, caravan_df_testLR$Purchase)
LRPerf <- performance(LRPred, "tpr", "fpr")

plotROC = 0
if (plotROC == 1) { # plots ROC
	dev.new()
	plot(LRPerf, colorize=TRUE)
    abline(a=0, b=1, lty=2, lwd=3, col="black")  
} 

#AUC
auc <- performance(LRPred, "auc")  

#Corresponding Performance Measures
LRPrediction <- factor(as.factor(LOutPredictedClass), c(0, 1), 
                       labels = c("Not Purchased", "Purchased"))
LRActual <- factor(as.factor(caravan_df_testLR$Purchase), 
                   c(0, 1), labels = c("Not Purchased", "Purchased"))
CMLR <- confusionMatrix(LRActual, LRPrediction, negative = "Not Purchased" )



# diagnosticErrors computes various diagnostic errors useful for evaluating 
# the performance of a diagnostic test or a classifier: 
# accuracy (acc), sensitivity (sens), specificity (spec), 
# positive predictive value (ppv), negative predictive value (npv), and log-odds ratio (lor).

# percentage of correct classification (Accuracy)
mean(LRPrediction ==LRActual)
diagnosticErrors(CMLR)  

# comparing models
normal.LRPerf <- LRPerf
normal.LRPred <- LRPred
normal.auc <- performance(LRPred, "auc")  
normal.diagnosticErrors <- diagnosticErrors(CMLR)  
normal.CMLR <- CMLR
normal.logisticTrainingFit <- summary(logisticTrainingFit)

# ########################################################
# 			-- LOGISTIC Undersampled --
# ########################################################

# ****************************************************** #
# Logistic Regression Using Undersampled Data
# In [221]:
# Under Sampling Data
# Taking all the observations with dependent variable = 1
# ****************************************************** #

caravan_df_train_LR_Under <- caravan_df_trainLR[caravan_df_trainLR$Purchase==1,]
length(caravan_df_train_LR_Under$Purchase)
#Randomly select observations with dependent variable = 0
zeroObs <- caravan_df_trainLR[caravan_df_trainLR$Purchase==0,]
set.seed(123457)

# SAMPLING SELECTION !!!!!
samplin_level = 1
samplin_level = 2.5


rearrangedZeroObs <-  zeroObs[sample(nrow(zeroObs), length(caravan_df_train_LR_Under$Purchase) * samplin_level ),]


#Appending rows of randomly selected 0s in our undersampled data frame
caravan_df_train_LR_Under <- rbind(caravan_df_train_LR_Under, rearrangedZeroObs)
length(caravan_df_train_LR_Under$Purchase)

#Let's verify that number of 1s and 0s in our undersampled data are equal
undersampled.Frequency.Purchase <- data.frame(Purchase = levels(as.factor(caravan_df_train_LR_Under$Purchase)), Count = as.numeric(table(caravan_df_train_LR_Under$Purchase)))
undersampled.Frequency.Purchase

#Classify Using Logistic Regression with Undersampling
logisticTrainingFit <- glm(Purchase ~ ., family = "binomial", data = caravan_df_train_LR_Under)

 
  
#Predict Class And Display Confusion Matrix
LOutPredicted <- predict(logisticTrainingFit, caravan_df_testLR, type = "response")
LOutPredictedClass <- ifelse(LOutPredicted>0.5, 1, 0)
LOutActual <- caravan_df_testLR$Purchase
LConfusionOutPredicted <- table(LOutActual, LOutPredictedClass)
rownames(LConfusionOutPredicted) <- c("0","1")
colnames(LConfusionOutPredicted) <- c("0","1")
LConfusionOutPredicted
for (i in 1:4000){
  probabilityDF$LRU[i] <- LOutPredicted[i]
}


# Plot ROC and AUC for LR
probs <- LOutPredicted
library(ROCR)
LRPred <- prediction(probs, caravan_df_testLR$Purchase)
LRPerf <- performance(LRPred, "tpr", "fpr")

if (plotROC == 1) { # plots ROC
	plot(LRPerf, colorize=TRUE)
    abline(a=0, b=1, lty=2, lwd=3, col="black")  
}

#AUC
performance(LRPred, "auc")

#Corresponding Performance Measures
LRPrediction <- factor(as.factor(LOutPredictedClass), c(0, 1), labels = c("Not Purchased", "Purchased"))
LRActual <- factor(as.factor(caravan_df_testLR$Purchase), c(0, 1), labels = c("Not Purchased", "Purchased"))
library(crossval)
CMLR <- confusionMatrix(LRActual, LRPrediction, negative = "Not Purchased" )
diagnosticErrors(CMLR)

# comparing models
usample.LRPerf <- LRPerf
usample.LRPred <- LRPred
usample.auc <- performance(LRPred, "auc")  
usample.diagnosticErrors <- diagnosticErrors(CMLR)  
usample.CMLR <- CMLR
usample.logisticTrainingFit <- summary(logisticTrainingFit)

# ########################################################
# 			-- LOGISTIC OVERsampled --
# ########################################################

# **************************************************************** #
# Logistic Regression Using Oversampling
# In [224]:
# **************************************************************** #

#First let's recall the ratio of 1s and 0s in our original training dataset
paste0("Ratio of 1s to 0s- 1:", as.numeric(table(caravan_df_trainLR$Purchase))[1]/as.numeric(table(caravan_df_trainLR$Purchase))[2])

#Now let's duplicate the observations with dependent variable = 1 to make the ratio approximately 1:2
caravan_df_train_LR_OverDummy <- caravan_df_trainLR[caravan_df_trainLR$Purchase==1,]
caravan_df_train_LR_Over <- NULL
# SAMPLING SELECTION !!!!!
samplin_level <- 7
samplin_level <- 6

for (i in 1:samplin_level){
  caravan_df_train_LR_Over <- rbind(caravan_df_train_LR_Over, caravan_df_train_LR_OverDummy)
}

caravan_df_train_LR_Over <- rbind(caravan_df_train_LR_Over, caravan_df_trainLR[caravan_df_trainLR$Purchase==0,])
#Let's verify the number of 1s and 0s in our oversampled data
oversampled.Frequency.Purchase <- data.frame(Purchase = levels(as.factor(caravan_df_train_LR_Over$Purchase)), Count = as.numeric(table(caravan_df_train_LR_Over$Purchase)))
oversampled.Frequency.Purchase

#Let's verify the ratio of 1s to 0s after over sampling
paste0("Ratio of 1s to 0s After Oversampling- 1:", as.numeric(table(caravan_df_train_LR_Over$Purchase))[1]/as.numeric(table(caravan_df_train_LR_Over$Purchase))[2])

#Classify Using Logistic Regression with Oversampling
logisticTrainingFit <- glm(Purchase ~ ., family = "binomial", data = caravan_df_train_LR_Over)


#Predict Class And Display Confusion Matrix
LOutPredicted <- predict(logisticTrainingFit, caravan_df_testLR, type = "response")
LOutPredictedClass <- ifelse(LOutPredicted>0.5, 1, 0)
LOutActual <- caravan_df_testLR$Purchase
LConfusionOutPredicted <- table(LOutActual, LOutPredictedClass)
rownames(LConfusionOutPredicted) <- c("0","1")
colnames(LConfusionOutPredicted) <- c("0","1")
LConfusionOutPredicted
for (i in 1:4000){
    probabilityDF$LRO[i] <- LOutPredicted[i]
    }
for (i in 1:4000){
    probabilityDF$Actual[i] <- LOutActual[i]
}


# Plot ROC and AUC for LR
probs <- LOutPredicted
library(ROCR)
LRPred <- prediction(probs, caravan_df_testLR$Purchase)
LRPerf <- performance(LRPred, "tpr", "fpr")

if (plotROC == 1) { # plots ROC
	plot(LRPerf, colorize=TRUE)
    abline(a=0, b=1, lty=2, lwd=3, col="black")  
}

#AUC
auc <- performance(LRPred, "auc")

#Corresponding Performance Measures
LRPrediction <- factor(as.factor(LOutPredictedClass), c(0, 1), labels = c("Not Purchased", "Purchased"))
LRActual <- factor(as.factor(caravan_df_testLR$Purchase), c(0, 1), labels = c("Not Purchased", "Purchased"))
library(crossval)
CMLR <- confusionMatrix(LRActual, LRPrediction, negative = "Not Purchased" )
diagnosticErrors(CMLR)

# comparing models
osample.LRPerf <- LRPerf
osample.LRPred <- LRPred
osample.auc <- performance(LRPred, "auc")  
osample.diagnosticErrors <- diagnosticErrors(CMLR)  
osample.CMLR <- CMLR
osample.logisticTrainingFit <- summary(logisticTrainingFit)

# save to excel
wb = createWorkbook()

sheet = createSheet(wb, "Normal LR")

addDataFrame(normal.diagnosticErrors, sheet=sheet, startRow = 1, startColumn=2, row.names=TRUE)
addDataFrame(normal.CMLR, sheet=sheet, startRow =10, startColumn=2, row.names=TRUE)

sheet = createSheet(wb, "Undersampled LR")

addDataFrame(usample.diagnosticErrors, sheet=sheet, startRow = 1, startColumn=2, row.names=TRUE)
addDataFrame(usample.CMLR, sheet=sheet, startRow =10, startColumn=2, row.names=TRUE)

sheet = createSheet(wb, "Oversampled LR")

addDataFrame(osample.diagnosticErrors, sheet=sheet, startRow = 1, startColumn=2, row.names=TRUE)
addDataFrame(osample.CMLR, sheet=sheet, startRow =10, startColumn=2, row.names=TRUE)

saveWorkbook(wb, "My_File.xlsx")


# PLOT ROC CURVE 
ppi<-300
png(filename="ROC curve.png", width=6*ppi, height=6*ppi,res=ppi)
plot(normal.LRPerf, col=2, main='ROC curve')
abline(a=0, b=1, lty=2, lwd=3, col="black")
legend(0.5,0.5, c('Actual data','Under sampled data','Over sampled data', ''), 2:5)
plot(usample.LRPerf, col=3, add=TRUE)
plot(osample.LRPerf, col=4, add=TRUE)

dev.off()


osample.logisticTrainingFit



