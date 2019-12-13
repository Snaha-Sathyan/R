#DATASET: https://www.kaggle.com/johnsmith88/heart-disease-dataset

#Loading Important Libraries
library(gmodels)
library(Amelia)
library(dummies)
library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(ROCR)

#Loading The Dataset
data=read.csv("C:/Users/Sneha/Desktop/Datasets/heart.csv")
data[1:4,]

#Changing the column names
colnames(data)=c("age","sex","chest_pain_type","rest_bps",
                 "chol","fbs","rest_ecg","max_heart_rate",
                 "ex_ang","ST_depr","slope_ST","fluoro_scan",
                 "thal","target")
data[1:3,]

#Dimension/shape/no.of rows & columns in the dataset
dim(data)

#Identifying non-numeric(factor)columns
str(data)

#Summary of the dataset viz max,min,NA's,etc
summary(data)

#Checking y-variable for bias
CrossTable(data$target)

##########################
# DATA PRE-PROCESSING    #
##########################

#Checking NA & Missing values
missmap(data)#plot
sum(is.na(data))

#Visualisation
featurePlot(x=data[,-14], y=data[,14], plot="pairs",
            auto.key=list(columns=3))

names(data)
#Converting some int/num columns into Factor columns
factor.vars=c('sex','chest_pain_type','fbs','rest_ecg',
              'ex_ang','slope_ST','fluoro_scan','thal')
data[factor.vars]=data.frame(apply(data[factor.vars],
                                   2,
                                   as.factor))
str(data)

#Creating dummies
data=dummy.data.frame(data,all = TRUE,sep="_")
#After dummy
dim(data)

#Converting y-variable(int) into factor
data$target=as.factor(data$target)

#Scaling
data[c('age','rest_bps','chol','max_heart_rate','ST_depr')] <- scale(data[c('age','rest_bps',
                                                                            'chol','max_heart_rate',
                                                                            'ST_depr')]) 
str(data)

###############
# MODELLING   #
###############

#Creating Training & Testing Sets
set.seed(100)
part_data=createDataPartition(y=data$target, p = 0.70, list = FALSE)
train_data=data[part_data,]
test_data=data[-part_data,]

dim(test_data)
dim(train_data)

CrossTable(train_data$target)
CrossTable(test_data$target)

#Fitting the Model

model=glm(target~.,data=train_data,family = binomial(link='logit'))
summary(model)

#prediction
pred=predict(model,test_data,type = 'response')
result_pred <- ifelse(pred > 0.5,1,0)

Accu <- mean(result_pred == test_data$target)
print(paste('Accuracy',Accu))

error=mean(result_pred != test_data$target)
print(paste('Error',error))

#Roc Curve
p <- predict(model,test_data, type="response")
pr <- prediction(p, test_data$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
plot(prf, type="S",col="red",lwd=3,
     main="ROC Curve",legacy.axes=TRUE,
     xlab="False Positive Percentage",
     ylab="True Positive Percentage",
     )
#Confusion Matrix
cm=table(result_pred,test_data$target)
cm




