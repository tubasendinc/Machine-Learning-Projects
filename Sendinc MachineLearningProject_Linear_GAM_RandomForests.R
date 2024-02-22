#Sendinc Machine Learning Project: Linear Model, GAM, and Random Forests


options(scipen = 10)
install.packages("mboost")
library(devtools); install_github("therneau/survival")
install.packages("mgcv")
install.packages("caret")
library(caret)
library(haven)
library(stargazer)
library(mlr)
library(mgcv)
#REPLICATION
rep_data <- read_dta("sj-dta-1-jcr-10.1177_0022002720957713.dta")

#main model Table 2 Model 1 in the original article
model1=lm(idealptdiff ~ matraineesmilper + polity2 + usalliance +gdppc, data=rep_data)
summary(model1)

coeftest(model1, vcov = vcovHC, cluster = ~ ccode)

#I am able to replicate the coefficients and standard errors as reported in the article
stargazer(coeftest(model1, vcov = vcovHC, cluster = ~ ccode))
#PREDICTIONS
#subsetting the data

data = subset(rep_data, select = c(idealptdiff,matraineesmilper,polity2,usalliance,gdppc))
data <- na.omit(data)
datad=as.data.frame(data)
stargazer(datad)

#creating train and test sets
set.seed(1234) #setting seed for reproducibility
dt = sort(sample(nrow(data), nrow(data)*.7))
train<-data[dt,]
test<-data[-dt,]
modelgam <- mgcv::gam(idealptdiff ~ matraineesmilper + polity2 + usalliance + gdppc,data = data)

#GAM Model 
#5 fold Cross Validation Using Training Data

set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 5, repeats = 5)

modelgam <- train(idealptdiff ~., data = train, method = "gam", 
               trControl = train.control)
print(modelgam) #mse 0.2566158

#out of sample predictions for gam using test data
predictions <- modelgam %>% predict(test)

par(mfrow=c(1,2))
#plot of predicted and actual values
plot(predictions,test$idealptdiff,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)

#residual plot
resid.gam = test$idealptdiff - predictions
plot(test$idealptdiff, resid.gam, 
     xlab="Ideal Point Difference",
     ylab="Residuals")

#Random Forest
task <- makeRegrTask(data=train, target = "idealptdiff")

#specifiying the random forest classifier
forest <- makeLearner("regr.randomForest")

forestParamSpace <- makeParamSet(
  makeIntegerParam("ntree", lower = 300, upper = 300),
  makeIntegerParam("mtry", lower = 1, upper = 4),
  makeIntegerParam("nodesize", lower = 1, upper = 10),
  makeIntegerParam("maxnodes", lower = 5, upper = 20))
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
tunedForestModel <- mlr::train(tunedForest, task)

#Cross validating the model building

outer <- makeResampleDesc("CV", iters = 5)
forestWrapper <- makeTuneWrapper("regr.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
cvWithTuningRandom <- resample(forestWrapper, task, resampling = outer)
parallelStop()
cvWithTuningRandom  #0.1846633

#making predictions
predsRF <- data.frame(predict(tunedForestModel, newdata = test))

par(mfrow=c(1,2))

#plot of predicted and actual values
plot(predsRF$response,test$idealptdiff,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)

#residual plot
resid.rf = test$idealptdiff - predsRF$response
plot(test$idealptdiff, resid.rf, 
     xlab="Ideal Point Difference",
     ylab="Residuals")

#Linear Model
set.seed(1234) #setting seed for reproducibility

modellm <- train(idealptdiff ~., data = train, method = "lm", 
                  trControl = train.control)
print(modellm)

#out of sample predictions for gam using test data
predictionslm <- modellm %>% predict(test)

par(mfrow=c(1,2))
#plot of predicted and actual values
plot(predictionslm,test$idealptdiff,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)

#residual plot
resid.lm = test$idealptdiff - predictionslm
plot(test$idealptdiff, resid.lm, 
     xlab="Ideal Point Difference",
     ylab="Residuals")

#Investigating Variable Importance. I use full data.

#FOR LINEAR MODEL using 5 fold cv with 5 iterations
#introducing variables one by one

set.seed(1234) #setting seed for reproducibility

modellm <- train(idealptdiff ~ matraineesmilper, data = data, method = "lm", 
                 trControl = train.control)
print(modellm)
#mse
(0.8406471)^2 #0.7066875

set.seed(1234) #setting seed for reproducibility

modellm <- train(idealptdiff ~ matraineesmilper + polity2, data = data, method = "lm", 
                 trControl = train.control)
print(modellm)
#mse
(0.6580383)^2 #0.4330144

set.seed(1234) #setting seed for reproducibility

modellm <- train(idealptdiff ~ matraineesmilper + polity2 + usalliance, data = data, method = "lm", 
                 trControl = train.control)
print(modellm)

#mse
(0.6023114)^2 #0.362779

set.seed(1234) #setting seed for reproducibility

modellm <- train(idealptdiff ~ matraineesmilper + polity2 + usalliance +gdppc, data = data, method = "lm", 
                 trControl = train.control)
print(modellm)

#mse
(0.549897 )^2 #0.3023867

#omitting one variable each time

#omitting gdppc
set.seed(1234) #setting seed for reproducibility

modellm <- train(idealptdiff ~ matraineesmilper + polity2 + usalliance , data = data, method = "lm", 
                 trControl = train.control)
print(modellm)

#mse
(0.6023114)^2 #0.362779

#omitting us alliance

set.seed(1234) #setting seed for reproducibility

modellm <- train(idealptdiff ~ matraineesmilper + polity2 +gdppc, data = data, method = "lm", 
                 trControl = train.control)
print(modellm)

#mse

(0.5900286)^2 #0.3481337

#omitting polity2
set.seed(1234) #setting seed for reproducibility

modellm <- train(idealptdiff ~ matraineesmilper + usalliance +gdppc, data = data, method = "lm", 
                 trControl = train.control)
print(modellm)

#mse
(0.6105179 )^2 #  0.3727321

#omitting matraineesmilper
set.seed(1234) #setting seed for reproducibility

modellm <- train(idealptdiff ~ polity2 + usalliance +gdppc, data = data, method = "lm", 
                 trControl = train.control)
print(modellm)

#mse

(0.550997)^2 # 0.3035977

#GAM MODEL using 5 fold cv with 5 iterations 

#introducing variables one by one
set.seed(1234) #setting seed for reproducibility

modelgam <- train(idealptdiff ~., data = data, method = "gam", 
                  trControl = train.control)
print(modelgam)

(0.5121207)^2 #0.2622676

set.seed(1234) #setting seed for reproducibility

modelgam <- train(idealptdiff ~matraineesmilper, data = data, method = "gam", 
                  trControl = train.control)
print(modelgam)

(0.8323055)^2 #0.6927324

set.seed(1234) #setting seed for reproducibility

modelgam <- train(idealptdiff ~matraineesmilper + polity2, data = data, method = "gam", 
                  trControl = train.control)
print(modelgam)

(0.5907091  )^2 #0.3489372

set.seed(1234) #setting seed for reproducibility

modelgam <- train(idealptdiff ~matraineesmilper + polity2 + usalliance, data = data, method = "gam", 
                  trControl = train.control)
print(modelgam)

(0.5534938)^2 #0.3063554

set.seed(1234) #setting seed for reproducibility

modelgam <- train(idealptdiff ~matraineesmilper + polity2 + usalliance + gdppc, data = data, method = "gam", 
                  trControl = train.control)
print(modelgam)

(0.5121207)^2 #0.2622676


#omitting one variable at a time

#omitting gdppc

set.seed(1234) #setting seed for reproducibility

modelgam <- train(idealptdiff ~matraineesmilper + polity2 + usalliance, data = data, method = "gam", 
                  trControl = train.control)
print(modelgam)

(0.5534938)^2 #0.3063554

#omitting usalliance

set.seed(1234) #setting seed for reproducibility

modelgam <- train(idealptdiff ~matraineesmilper + polity2 + gdppc, data = data, method = "gam", 
                  trControl = train.control)
print(modelgam)

(0.5510628)^2 #0.3036702

#omitting polity2

set.seed(1234) #setting seed for reproducibility

modelgam <- train(idealptdiff ~matraineesmilper  + usalliance + gdppc, data = data, method = "gam", 
                  trControl = train.control)
print(modelgam)

(0.5726987)^2 #0.3279838

#omitting matraineesmilper

set.seed(1234) #setting seed for reproducibility

modelgam <- train(idealptdiff ~  polity2 + usalliance + gdppc, data = data, method = "gam", 
                  trControl = train.control)
print(modelgam)

(0.5127227)^2 # 0.2628846

#Random Forests 

#introducing variables one by one

#for random forest 
#adding variables one by one

taskfull <- makeRegrTask(data=data[1:2], target = "idealptdiff")
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = taskfull,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()
forestWrapper <- makeTuneWrapper("regr.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
RFmsep <- resample(forestWrapper, taskfull, resampling = outer)
parallelStop()

RFmsep #0.6313493

#############
taskfull <- makeRegrTask(data=data[1:3], target = "idealptdiff")
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = taskfull,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()
forestWrapper <- makeTuneWrapper("regr.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
RFmsep <- resample(forestWrapper, taskfull, resampling = outer)
parallelStop()

RFmsep #0.3055535

#############
taskfull <- makeRegrTask(data=data[1:4], target = "idealptdiff")
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = taskfull,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()
forestWrapper <- makeTuneWrapper("regr.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
RFmsep <- resample(forestWrapper, taskfull, resampling = outer)
parallelStop()

RFmsep #0.2589227

#############
taskfull <- makeRegrTask(data=data[1:5], target = "idealptdiff")
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = taskfull,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()
forestWrapper <- makeTuneWrapper("regr.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
RFmsep <- resample(forestWrapper, taskfull, resampling = outer)
parallelStop()

RFmsep #0.1835790

#omitting variables one by one

#omitting matraineesmilper
taskfull <- makeRegrTask(data=data[-2], target = "idealptdiff")
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = taskfull,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()
forestWrapper <- makeTuneWrapper("regr.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
RFmsep <- resample(forestWrapper, taskfull, resampling = outer)
parallelStop()

RFmsep #0.2032006

#omitting polity2
taskfull <- makeRegrTask(data=data[-3], target = "idealptdiff")
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = taskfull,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()
forestWrapper <- makeTuneWrapper("regr.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
RFmsep <- resample(forestWrapper, taskfull, resampling = outer)
parallelStop()

RFmsep #0.2092798

#omitting usalliance
taskfull <- makeRegrTask(data=data[-4], target = "idealptdiff")
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = taskfull,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()
forestWrapper <- makeTuneWrapper("regr.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
RFmsep <- resample(forestWrapper, taskfull, resampling = outer)
parallelStop()

RFmsep #0.2126090

#omitting gdppc
taskfull <- makeRegrTask(data=data[-5], target = "idealptdiff")
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = taskfull,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()
forestWrapper <- makeTuneWrapper("regr.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)
set.seed(123) #setting seed for reproducibility
parallelStartSocket(cpus = detectCores())
RFmsep <- resample(forestWrapper, taskfull, resampling = outer)
parallelStop()

RFmsep #0.2126090

#specifying non-linear relationships with GAM

par(mfrow=c(2,2))

modelgam1 <- mgcv::gam(idealptdiff ~ s(matraineesmilper) +polity2 + usalliance+ gdppc,data = data)
plot(modelgam1)

modelgam2 <- mgcv::gam(idealptdiff ~ matraineesmilper + s(polity2) + usalliance+ gdppc,data = data)
plot(modelgam2)

modelgam3 <- mgcv::gam(idealptdiff ~ matraineesmilper + polity2 + usalliance + s(gdppc),data = data)
plot(modelgam3)

modelgam <- mgcv::gam(idealptdiff ~ s(matraineesmilper, bs='ps', sp=0.2) + s(polity2, bs='ps', sp=0.6) + s(usalliance, bs='ps', sp=0.6) + s(gdppc, bs='ps', sp=0.6),data = data)
plot(modelgam)


