#Project: Identifying Presidential Threats Using Naive Bayes, Random Forests, and SVM

install.packages("readtext")
install.packages("splitstackshape")
install.packages("stringr")
install.packages("writexl")
install.packages("e1071")
install.packages("gmodels")
install.packages("caTools")
install.packages("SnowballC")
library(splitstackshape)
library(parallelMap)
library(parallel)
library(readtext)
library(lmtest)
library(sandwich)
library(mlr)
library(MLmetrics)
library(xgboost)
library(mlbench)
library(kernlab)
library(readr)
library(parallelMap)
library(parallel)
library(e1071)

#PREDICTING THREATS
#creating the corpus
#remove special characters
library(readxl)
training_statements <- read_xls("training_statements_extended720.xls")

#recoding NAs
training_statements$threat[is.na(training_statements$threat)]=0

#making factor
training_statements$threat <- factor(training_statements$threat)


#removing special characters
library(stringr)
training_statements$text2=str_replace_all(training_statements$text2, "[^[:alnum:]]", " ")

training_statements$text2=gsub("[^\u0001-\u007F]+|<U\\+\\w+>","", training_statements$text2)
#create a corpus
library(tm)
corpus <- VCorpus(VectorSource(training_statements$text2))

#lower case
corpus_clean <- tm_map(corpus, content_transformer(tolower))

#remove numbers
corpus_clean <- tm_map(corpus_clean, removeNumbers)

#remove punctuation
corpus_clean <- tm_map(corpus_clean, removePunctuation)

#remove stop words
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())

#stemming
#library(SnowballC)

#corpus_clean <- tm_map(corpus_clean, stemDocument)

#create document term matrix
dtm <- DocumentTermMatrix(corpus_clean)

#removing words that appear less than 13 times

freq_words <- findFreqTerms(dtm, 13)

dtm <- dtm[ , freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

dtm <- apply(dtm, MARGIN = 2, convert_counts)

#create a data frame from document term matrix
statements_data = as.data.frame(as.matrix(dtm))
colnames(statements_data) = make.names(colnames(statements_data))
statements_data$threat = training_statements$threat

names <- c(1:1363)
statements_data[,names] <- lapply(statements_data[,names] , factor)

#split into train and test sets 70% training and 30% test set
statement_train <- statements_data[1:216, ]
statement_test <- statements_data[217:720, ]

#specifiying the task. I use this task in training all of the models
library(mlr)
task <- makeClassifTask(data=statement_train, target = "threat")
tasknb2 <- makeClassifTask(data=statements_data, target = "threat")

#learner for naive bayes
bayes <- makeLearner("classif.naiveBayes", predict.threshold = .5,predict.type = "prob")

#training the model
NBModel <- train(bayes, task)

kFold <- makeResampleDesc(method = "RepCV", folds = 8, reps = 10,
                          stratify = TRUE)
set.seed(123) #setting seed for reproducibility

bayesCV <- resample(learner = bayes, task = task,
                    resampling = kFold,
                    measures = list(mmce, acc, fpr, fnr))
#acc.test.mean=0.9435065
#calculating the predicted probabilities in the test data
predsNB <- data.frame(predict(NBModel, newdata = statement_test,type="class"))

#looking at confusion matrix to evaluate the performance of the NB algorithm 
confusionmatrixNB <- table(predsNB$truth, predsNB$response)
prop.table(confusionmatrixNB,1) 
AUCNB=AUC(y_pred = predsNB$response,y_true=predsNB$truth)
PRAUCNB=PRAUC(y_pred = as.numeric(predsNB$response),y_true=as.numeric(predsNB$truth))

#changing the laplace smoothing
NBlaplace <- naiveBayes(threat ~., data=statement_train, laplace = 10,threshold=.2)
laplacepred <- predict(NBlaplace, newdata = statement_test,type="class")
confusionmatrixlaplace <-table( statement_test$threat,laplacepred, dnn=c("Actual","Prediction"))
confusionmatrixlaplace
AUC2=AUC(y_pred = laplacepred,y_true=statement_test$threat)
PRAUC2=PRAUC(y_pred = as.numeric(laplacepred),y_true=statement_test$threat)


#############
#RANDOM FOREST
#####PREDICTING THREATS
dtm <- DocumentTermMatrix(corpus_clean)

freq_words <- findFreqTerms(dtm, 10)

dtm <- dtm[ , freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

dtm <- apply(dtm, MARGIN = 2, convert_counts)

statements_data = as.data.frame(as.matrix(dtm))
colnames(statements_data) = make.names(colnames(statements_data))
statements_data$threat = training_statements$threat

names <- c(1:1696)
statements_data[,names] <- lapply(statements_data[,names] , factor)

#split into train and test sets 70% training and 30% test set
statement_train <- statements_data[1:216, ]
statement_test <- statements_data[217:720, ]

task <- makeClassifTask(data=statement_train, target = "threat",fixup.data = "no",check.data = FALSE)

#specifiying the random forest classifier
forest <- makeLearner("classif.randomForest",predict.threshold = .4,predict.type = "prob")

forestParamSpace <- makeParamSet(
  makeIntegerParam("ntree", lower = 300, upper = 800),
  makeIntegerParam("mtry", lower = 150, upper = 1696),
  makeIntegerParam("nodesize", lower = 1, upper = 200),
  makeIntegerParam("maxnodes", lower = 5, upper = 200))
set.seed(123) #setting seed for reproducibility

randSearch <- makeTuneControlRandom(maxit = 100)

cvForTuning <- makeResampleDesc("CV", iters = 5)

set.seed(123) #setting seed for reproducibility

parallelStartSocket(cpus = detectCores())

tunedForestPars <- tuneParams(forest, task = task,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)

parallelStop()

tunedForestPars

#training the final model
tunedForest <- setHyperPars(forest, par.vals = tunedForestPars$x)
tunedForestModel <- train(tunedForest, task)

#Cross validating the model building

outer <- makeResampleDesc("CV", iters = 5)
forestWrapper <- makeTuneWrapper("classif.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
cvWithTuningRandom <- resample(forestWrapper, task, resampling = outer)
parallelStop()
cvWithTuningRandom 

predsRF <- data.frame(predict(tunedForestModel, newdata = statement_test))

confusionmatrix <-table(predsRF$truth, predsRF$response,dnn=c("Actual","Prediction"))
confusionmatrix

AUCRF=AUC(y_pred = predsRF$response,y_true=predsRF$truth)
PRAUCRF=PRAUC(y_pred = as.numeric(predsRF$response),y_true=as.numeric(predsRF$truth))


#####SVM
#PREDICTING THREATS
library(mlr)
task <- makeClassifTask(data=statement_train, target = "threat",fixup.data = "no",check.data = FALSE)
tasksvm2 <- makeClassifTask(data=statements_data, target = "threat")

#specifiying the SVM classifier
svmclassif <- makeLearner("classif.svm",predict.threshold = .5,predict.type = "prob")

#defining the hyperparameter space for tuning for svm
kernels <- c("polynomial", "radial", "sigmoid")

svmParamSpace <- makeParamSet(
  makeDiscreteParam("kernel", values = kernels),
  makeIntegerParam("degree", lower = 1, upper = 3),
  makeNumericParam("cost", lower = 0.1, upper = 10),
  makeNumericParam("gamma", lower = 0.1, 10))

#defining random search
set.seed(123) #setting seed for reproducibility

randSearch <- makeTuneControlRandom(maxit = 20)

#holdout CV with 2/3 split
cvForTuning <- makeResampleDesc("Holdout", split = 2/3)

#performing hyperparameter tuning
set.seed(123) #setting seed for reproducibility

parallelStartSocket(cpus = detectCores())
tunedSvmPars <- tuneParams("classif.svm", task = task,
                           resampling = cvForTuning,
                           par.set = svmParamSpace,
                           control = randSearch)

parallelStop()

#extracting the winning hyperparameter values from tuning
tunedSvmPars
tunedSvmPars$x

#building the model with the best performing combination, i.e., 2nd degree polynomial function
tunedSvm <- setHyperPars(makeLearner("classif.svm"),
                         par.vals = tunedSvmPars$x)

#training the model with the tuned hyperparameter
tunedSvmModel <- train(tunedSvm, task)

#cross-validating the model-building process

#define CV strategy. 3-fold CV.
outer <- makeResampleDesc("CV", iters = 5)

#make wrapped learner
set.seed(123) #setting seed for reproducibility
svmWrapper <- makeTuneWrapper("classif.svm", resampling = cvForTuning,
                              par.set = svmParamSpace,
                              control = randSearch)
#nested CV
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
cvWithTuningSVM <- resample(svmWrapper, task, resampling = outer)
parallelStop()

#extracting the cross-validation result
cvWithTuningSVM

predsSVM <- data.frame(predict(tunedSvmModel, newdata = statement_test))

confusionmatrixSVM <- table(predsSVM$truth, predsSVM$response)
confusionmatrixSVM

AUCSVM=AUC(y_pred = predsSVM$response,y_true=predsSVM$truth)
PRAUCSVM=PRAUC(y_pred = as.numeric(predsSVM$response),y_true=as.numeric(predsSVM$truth))


###############
#PARAGRAPH LEVEL
###############

#PREDICTING THREATS
#creating the corpus
#remove special characters
library(readxl)
training_statements <- read_xls("training_paragraph level.xls")

#recoding NAs
training_statements$threat[is.na(training_statements$threat)]=0
training_statements$threat[(training_statements$threat)=="NA"]=0

#making factor
training_statements$threat <- factor(training_statements$threat)

#removing special characters
library(stringr)
training_statements$text2.y=str_replace_all(training_statements$text2.y, "[^[:alnum:]]", " ")

training_statements$text2.y=gsub("[^\u0001-\u007F]+|<U\\+\\w+>","", training_statements$text2.y)
#create a corpus
library(tm)
corpus <- VCorpus(VectorSource(training_statements$text2.y))

#lower case
corpus_clean <- tm_map(corpus, content_transformer(tolower))

#remove numbers
corpus_clean <- tm_map(corpus_clean, removeNumbers)

#remove punctuation
corpus_clean <- tm_map(corpus_clean, removePunctuation)

#remove stop words
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())

#stemming
#library(SnowballC)

#corpus_clean <- tm_map(corpus_clean, stemDocument)

#create document term matrix
dtm <- DocumentTermMatrix(corpus_clean)

#removing words that appear less than 13 times

freq_words <- findFreqTerms(dtm, 13)

dtm <- dtm[ , freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

dtm <- apply(dtm, MARGIN = 2, convert_counts)

#create a data frame from document term matrix
statements_data = as.data.frame(as.matrix(dtm))
colnames(statements_data) = make.names(colnames(statements_data))
statements_data$threat = training_statements$threat

names <- c(1:1348)
statements_data[,names] <- lapply(statements_data[,names] , factor)

#split into train and test sets 70% training and 30% test set
statement_train <- statements_data[1:837, ]
statement_test <- statements_data[838:2792, ]

#specifiying the task. I use this task in training all of the models
library(mlr)
task <- makeClassifTask(data=statement_train, target = "threat")
tasknb2 <- makeClassifTask(data=statements_data, target = "threat")

#learner for naive bayes
bayes <- makeLearner("classif.naiveBayes", predict.threshold = .5,predict.type = "prob")

#training the model
NBModel <- train(bayes, task)

kFold <- makeResampleDesc(method = "RepCV", folds = 8, reps = 10,
                          stratify = TRUE)
set.seed(123) #setting seed for reproducibility

bayesCV <- resample(learner = bayes, task = task,
                    resampling = kFold,
                    measures = list(mmce, acc, fpr, fnr))
#acc.test.mean=0.9435065
#calculating the predicted probabilities in the test data
predsNB <- data.frame(predict(NBModel, newdata = statement_test,type="class"))

#looking at confusion matrix to evaluate the performance of the NB algorithm 
confusionmatrixNB <- table(predsNB$truth, predsNB$response)
prop.table(confusionmatrixNB,1) 
AUCNB=AUC(y_pred = predsNB$response,y_true=predsNB$truth)
PRAUCNB=PRAUC(y_pred = as.numeric(predsNB$response),y_true=as.numeric(predsNB$truth))

#changing the laplace smoothing
NBlaplace <- naiveBayes(threat ~., data=statement_train, laplace = 10,threshold=.3)
laplacepred <- predict(NBlaplace, newdata = statement_test,type="class")
confusionmatrixlaplace <-table( statement_test$threat,laplacepred, dnn=c("Actual","Prediction"))
confusionmatrixlaplace
AUC2=AUC(y_pred = laplacepred,y_true=statement_test$threat)
PRAUC2=PRAUC(y_pred = as.numeric(laplacepred),y_true=statement_test$threat)

#############
#RANDOM FOREST
#####PREDICTING THREATS
dtm <- DocumentTermMatrix(corpus_clean)

freq_words <- findFreqTerms(dtm, 10)

dtm <- dtm[ , freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

dtm <- apply(dtm, MARGIN = 2, convert_counts)

statements_data = as.data.frame(as.matrix(dtm))
colnames(statements_data) = make.names(colnames(statements_data))
statements_data$threat = training_statements$threat

names <- c(1:1681)
statements_data[,names] <- lapply(statements_data[,names] , factor)

#split into train and test sets 70% training and 30% test set
statement_train <- statements_data[1:837, ]
statement_test <- statements_data[838:2792, ]

task <- makeClassifTask(data=statement_train, target = "threat",fixup.data = "no",check.data = FALSE)

#specifiying the random forest classifier
forest <- makeLearner("classif.randomForest",predict.threshold = .4,predict.type = "prob")

forestParamSpace <- makeParamSet(
  makeIntegerParam("ntree", lower = 300, upper = 800),
  makeIntegerParam("mtry", lower = 150, upper = 800),
  makeIntegerParam("nodesize", lower = 1, upper = 200),
  makeIntegerParam("maxnodes", lower = 5, upper = 200))
set.seed(123) #setting seed for reproducibility

randSearch <- makeTuneControlRandom(maxit = 100)

cvForTuning <- makeResampleDesc("CV", iters = 5)

set.seed(123) #setting seed for reproducibility

parallelStartSocket(cpus = detectCores())

tunedForestPars <- tuneParams(forest, task = task,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)

parallelStop()

tunedForestPars

#training the final model
tunedForest <- setHyperPars(forest, par.vals = tunedForestPars$x)
tunedForestModel <- train(tunedForest, task)

#Cross validating the model building

outer <- makeResampleDesc("CV", iters = 5)
forestWrapper <- makeTuneWrapper("classif.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
cvWithTuningRandom <- resample(forestWrapper, task, resampling = outer)
parallelStop()
cvWithTuningRandom 

predsRF <- data.frame(predict(tunedForestModel, newdata = statement_test))

confusionmatrix <-table(predsRF$truth, predsRF$response,dnn=c("Actual","Prediction"))
confusionmatrix

AUCRF=AUC(y_pred = predsRF$response,y_true=predsRF$truth)
PRAUCRF=PRAUC(y_pred = as.numeric(predsRF$response),y_true=as.numeric(predsRF$truth))


#####SVM
#PREDICTING THREATS
library(mlr)
task <- makeClassifTask(data=statement_train, target = "threat",fixup.data = "no",check.data = FALSE)
tasksvm2 <- makeClassifTask(data=statements_data, target = "threat")

#specifiying the SVM classifier
svmclassif <- makeLearner("classif.svm",predict.threshold = .5,predict.type = "prob")

#defining the hyperparameter space for tuning for svm
kernels <- c("polynomial", "radial", "sigmoid")

svmParamSpace <- makeParamSet(
  makeDiscreteParam("kernel", values = kernels),
  makeIntegerParam("degree", lower = 1, upper = 3),
  makeNumericParam("cost", lower = 0.1, upper = 10),
  makeNumericParam("gamma", lower = 0.1, 10))

#defining random search
set.seed(123) #setting seed for reproducibility

randSearch <- makeTuneControlRandom(maxit = 20)

#holdout CV with 2/3 split
cvForTuning <- makeResampleDesc("Holdout", split = 2/3)

#performing hyperparameter tuning
set.seed(123) #setting seed for reproducibility

parallelStartSocket(cpus = detectCores())
tunedSvmPars <- tuneParams("classif.svm", task = task,
                           resampling = cvForTuning,
                           par.set = svmParamSpace,
                           control = randSearch)

parallelStop()

#extracting the winning hyperparameter values from tuning
tunedSvmPars
tunedSvmPars$x

#building the model with the best performing combination, i.e., 2nd degree polynomial function
tunedSvm <- setHyperPars(makeLearner("classif.svm"),
                         par.vals = tunedSvmPars$x)

#training the model with the tuned hyperparameter
tunedSvmModel <- train(tunedSvm, task)

#cross-validating the model-building process

#define CV strategy. 3-fold CV.
outer <- makeResampleDesc("CV", iters = 5)

#make wrapped learner
set.seed(123) #setting seed for reproducibility
svmWrapper <- makeTuneWrapper("classif.svm", resampling = cvForTuning,
                              par.set = svmParamSpace,
                              control = randSearch)
#nested CV
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
cvWithTuningSVM <- resample(svmWrapper, task, resampling = outer)
parallelStop()

#extracting the cross-validation result
cvWithTuningSVM

predsSVM <- data.frame(predict(tunedSvmModel, newdata = statement_test))

confusionmatrixSVM <- table(predsSVM$truth, predsSVM$response)
confusionmatrixSVM

AUCSVM=AUC(y_pred = predsSVM$response,y_true=predsSVM$truth)
PRAUCSVM=PRAUC(y_pred = as.numeric(predsSVM$response),y_true=as.numeric(predsSVM$truth))


