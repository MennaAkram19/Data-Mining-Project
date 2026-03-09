#############################################
# Data Mining Assignment
# Breast Cancer Classification using k-NN
# Student: Menna Akram
#############################################

############################
# Part 0: Install Libraries
############################
library(kknn)
library(caret)
library(ggplot2)


############################
# Part 1: Data Exploration
############################
data <- read.csv("wdbc.data", header = FALSE)

colnames(data) <- c("ID","Diagnosis",
                    
                    "Radius_mean","Texture_mean","Perimeter_mean","Area_mean","Smoothness_mean",
                    "Compactness_mean","Concavity_mean","ConcavePoints_mean","Symmetry_mean","FractalDimension_mean",
                    
                    "Radius_se","Texture_se","Perimeter_se","Area_se","Smoothness_se",
                    "Compactness_se","Concavity_se","ConcavePoints_se","Symmetry_se","FractalDimension_se",
                    
                    "Radius_worst","Texture_worst","Perimeter_worst","Area_worst","Smoothness_worst",
                    "Compactness_worst","Concavity_worst","ConcavePoints_worst","Symmetry_worst","FractalDimension_worst")


head(data)

str(data)

summary(data)


############################
#Missing Values
############################

sum(is.na(data))


############################
# Part 1.2 Data Quality
############################

table(data$Diagnosis)


range(data$Radius_mean)

range(data$Texture_mean)

range(data$Area_mean)


############################
# Boxplots
############################

boxplot(data$Radius_mean,
        col="gold",
        main="Radius Mean Distribution",
        ylab="Radius Mean")

boxplot(data$Texture_mean,
        col="skyblue",
        main="Texture Mean Distribution",
        ylab="Texture Mean")

boxplot(data$Area_mean,
        col="lightgreen",
        main="Area Mean Distribution",
        ylab="Area Mean")


############################
# Part 1.3 Data Visualization
############################
diagnosis_counts <- table(data$Diagnosis)

barplot(diagnosis_counts,
        col=c("mediumseagreen","tomato"),
        main="Distribution of Diagnosis",
        ylab="Number of Cases",
        xlab="Diagnosis")


############################
# Scatter Plot
############################

plot(data$Radius_mean,
     data$Texture_mean,
     col=ifelse(data$Diagnosis=="B","deepskyblue","red3"),
     pch=19,
     main="Radius vs Texture by Diagnosis",
     xlab="Radius Mean",
     ylab="Texture Mean")

legend("topright",
       legend=c("Benign","Malignant"),
       col=c("deepskyblue","red3"),
       pch=19)


################################
# Part 2: Data Preparation
################################

data$Diagnosis <- as.factor(data$Diagnosis)

data$ID <- NULL


################################
# Part 2.2 Train Test Split
################################

# لضمان تكرار نفس النتيجة

set.seed(123)


# تقسيم الداتا

train_index <- createDataPartition(data$Diagnosis,
                                   p=0.7,
                                   list=FALSE)


train_data <- data[train_index,]

test_data <- data[-train_index,]


# عرض حجم الداتا

cat("Training Set Size:", nrow(train_data),"\n")

cat("Testing Set Size:", nrow(test_data),"\n")


################################
# Part 2.3 Train kNN Model
################################

knn_model <- kknn(Diagnosis ~ .,
                  train=train_data,
                  test=test_data,
                  k=5,
                  kernel="rectangular")



summary(knn_model)


################################
# Part 2.4 Predictions
################################

predictions <- fitted(knn_model)

predictions <- as.factor(predictions)




head(predictions,10)


################################
# Part 3: Model Evaluation
################################

# Confusion Matrix

conf_matrix <- table(Actual=test_data$Diagnosis,
                     Predicted=predictions)

conf_matrix



TP <- conf_matrix["M","M"]
TN <- conf_matrix["B","B"]
FP <- conf_matrix["B","M"]
FN <- conf_matrix["M","B"]


################################
# Accuracy
################################

accuracy <- (TP+TN)/sum(conf_matrix)


################################
# Precision
################################

precision <- TP/(TP+FP)


################################
# Recall
################################

recall <- TP/(TP+FN)


################################
# F1 Score
################################

f1 <- 2*((precision*recall)/(precision+recall))


################################
# طباعة النتائج
################################

cat("Model Performance Metrics\n")

cat("Accuracy:", round(accuracy,4),"\n")

cat("Precision:", round(precision,4),"\n")

cat("Recall:", round(recall,4),"\n")

cat("F1 Score:", round(f1,4),"\n")