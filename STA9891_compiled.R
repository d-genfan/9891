rm(list = ls())    #delete objects
cat("\014")
library(class)
library(ggplot2)
library(dplyr)
library(doMC) # parallel computing
library(tidyverse)
library(glmnet)
library(ISLR)
library(readr)
library(randomForest)
library(visdat) 
library(AUC)
library(tm) # for stopwords
library(ggthemes)
library(latex2exp)
library(comprehenr)
library(gridExtra)

# link to dataset: https://www.kaggle.com/balaka18/email-spam-classification-dataset-csv/code
getwd();setwd("C:/Users/davin/OneDrive/Desktop/rWD/9891_PROJECT")
spam <- read.csv('emails.csv', header = T)
#spam  <- read.csv('~/Desktop/Baruch MS/FALL_2021/STA_9891/FinalProject/take_two/emails.csv', header = TRUE)
spam  <- spam[,-1] # Dropping Email.No. column

# Removing stop words from columns used as features
`%!in%` <- Negate(`%in%`)
new_spam <- spam[ ,which((colnames(spam) %!in% stopwords("SMART")) == TRUE)]
# Removing features that contain period punctuation "." in the name 
clean_spam <- new_spam[ , -c(contains(vars = colnames(new_spam), match = c(".")))]


# EDA visualization
# word count distribution
spam_col_Sums <- data.frame(matrix(ncol = 2, nrow = ncol(clean_spam)))
spam_col_Sums <- data.frame(word=spam_col_Sums$X1,count=spam_col_Sums$X2)
spam_col_Sums$word <- colnames(clean_spam)
spam_col_Sums$count <- colSums(clean_spam)
wc_distribution.plot <- ggplot(spam_col_Sums, aes(x = count)) +
  geom_histogram(colour = 'white', fill = 'firebrick3', bins = 50) +
  ggtitle("Distribution of word counts") + xlab("Word frequencies (log10 scaled)") +
  ylab("Count") + scale_x_log10()
wc_distribution.plot
ggsave("wc_distroplot.png", w = 6, h = 6, dpi = 300)

emailwc_distribution.plot <- ggplot(clean_spam, 
                                    aes(x = rowSums(clean_spam),
                                        fill = factor(to_vec(for(i in Prediction) if (i == 1) 'spam' else 'no spam')))) +
  geom_histogram(bins = 50, alpha = 0.6) + ylab("Count") + scale_x_log10() + guides(fill= guide_legend(title = NULL)) +
  scale_fill_manual(values = c('cyan3', 'coral2')) +
  ggtitle("Distribution word count per email") + xlab("Email word total (log10 scaled)")
emailwc_distribution.plot
ggsave("emailwc_distroplot.png", w = 6, h = 4, dpi = 320)

# Using full dataset for model
X = model.matrix(Prediction ~., data = clean_spam)
dim(X)
y = as.numeric(clean_spam$Prediction)
table(y)

Reps    = 50
p       = dim(X)[2]
n       = dim(X)[1]
n.train = floor(0.9*n)
n.test  = n-n.train
dt      = 0.01
thta    = 1-seq(0,1, by=dt)
thta.length = length(thta)

FPR.train_matrix <- matrix(0, nrow = thta.length, ncol = Reps)
TPR.train_matrix <- matrix(0, nrow = thta.length, ncol = Reps)
FPR.test_matrix <- matrix(0, nrow = thta.length, ncol = Reps)
TPR.test_matrix <- matrix(0, nrow = thta.length, ncol = Reps)

train_test_func <- function(x) {
  # shuffles indexs and returns indices for training set
  # and test set
  shuffled = sample(x)
  train = shuffled[1:n.train]
  test  = shuffled[(1+n.train):n]
  
  return(list(train = train, test = test))
}
tt_split <- function(i) {
  # uses shuffled indices to subselect rows from X.matrix and 
  # return list containing X.train, y.train, X.test, y.test
  ttf     = train_test_func(i)
  X.train = X[ttf$train,]
  y.train = y[ttf$train ]
  X.test  = X[ttf$test, ]
  y.test  = y[ttf$test  ]
  
  # weighting 
  not.spam = y.train[y.train==0]
  pos.spam = y.train[y.train==1]
  neg.weight = length(pos.spam)/length(not.spam)
  pos.weight = 1
  wt = ifelse(y.train == 0, neg.weight, 1)
  
  return(list(X.train = X.train, y.train = y.train, X.test = X.test, y.test = y.test, wt = wt))
}
# function to extract intercept value after glmnet() is fit
lasso.a0_tag       <- function(mod) mod$a0
# function to extract beta coefficient values after glmnet() is fit
lasso.betaHat_tag  <- function(mod) as.vector(mod$beta)
# function to extract lambda.min value once cv.glmnet is run
lambda_min <- function(mod) mod$lambda.min
get_lambda <- function(r){
  # executes cv.glmnet fit, then applies lambdd.min() to get lambda.min
  lambda_min(cv.glmnet(r$X.train, r$y.train, family = "binomial",alpha = 1,
                       type.measure="auc", intercept = TRUE, weights = r$wt, parallel = TRUE))
}
get_lasso0         <- function(r, lam) {
  # fits lasso model and returns intercept
  lasso.a0_tag(glmnet(r$X.train, r$y.train,lambda = lam, alpha = 1,
                      family = "binomial",intercept = TRUE, weights = r$wt, parallel = TRUE))
}
get_betaHat        <- function(r, lam) {
  # fits lasso model and returns beta coefficients as a vector
  lasso.betaHat_tag(glmnet(r$X.train, r$y.train,lambda = lam, alpha = 1,
                           family = "binomial",intercept = TRUE, weights = r$wt, parallel = TRUE))
}


# Generate 50 train/test splits of X.train, y.train
# and X.test, y.test
tt_reps <- replicate(50, lapply(n, FUN = tt_split)) # returns list of lists

# register cores for parallel computing
registerDoMC(cores = 2)

system.time(lambda.mins_lasso <- lapply(tt_reps, get_lambda)) #  runtime ~ 72 mins 
system.time(lasso0map <- pmap(list(tt_reps, lambda.mins_lasso) , get_lasso0)) # runtime ~13mins
system.time(lassobetamap   <- pmap(list(tt_reps, lambda.mins_lasso) , get_betaHat)) # runtime ~13mins

for (r in seq(50)) {
  X.train = tt_reps[[r]]$X.train
  y.train = tt_reps[[r]]$y.train
  X.test  = tt_reps[[r]]$X.test
  y.test  = tt_reps[[r]]$y.test
  lasso0.hat     = lasso0map[[r]]
  lasso.beta.hat = lassobetamap[[r]]
  
  for (i in c(1:thta.length)){
    prob.train     = exp(X.train %*% lasso.beta.hat + lasso0.hat)/(1 + exp(X.train %*% lasso.beta.hat +  lasso0.hat))
    prob.test      = exp(X.test %*% lasso.beta.hat + lasso0.hat)/(1 + exp(X.test %*% lasso.beta.hat +  lasso0.hat))
    P.train        = sum(y.train==1) # total positives in the data
    N.train        = sum(y.train==0) # total negatives in the data
    
    # calculate the FPR and TPR for train data 
    y.hat.train           = ifelse(prob.train > thta[i], 1, 0) # table(y.hat.train, y.train)
    FP.train              = sum(y.train[y.hat.train==1] == 0) # false positives 
    TP.train              = sum(y.hat.train[y.train==1] == 1) # true positives 
    FPR.train_matrix[i,r] = FP.train/N.train # FPR = type 1 error = 1 - specificity
    TPR.train_matrix[i,r] = TP.train/P.train # TPR = 1 - type 2 error = sensitivity
    
    # calculate the FPR and TPR for test data 
    y.hat.test           = ifelse(prob.test > thta[i], 1, 0)
    FP.test              = sum(y.test[y.hat.test==1] == 0) 
    TP.test              = sum(y.hat.test[y.test==1] == 1) 
    P.test               = sum(y.test==1) 
    N.test               = sum(y.test==0) 
    FPR.test_matrix[i,r] = FP.test/N.test 
    TPR.test_matrix[i,r] = TP.test/P.test
  }
}


# AUC matrices
auc_train_matrix <- matrix(0, Reps)
auc_test_matrix <- matrix(0, Reps)

# recording AUC scores for train and test sets for each repetition
for (i in c(1:Reps)) {
  auc_train_matrix[i]  <- sum((TPR.train_matrix[1:(thta.length-1),i] + 0.5 * 
                                 diff(TPR.train_matrix[,i])) * diff(FPR.train_matrix[,i]))
  auc_test_matrix[i]   <- sum((TPR.test_matrix[1:(thta.length-1),i] + 0.5 * 
                                 diff(TPR.test_matrix[,i])) * diff(FPR.test_matrix[,i]))
}

auc.train.df <- as.data.frame(cbind(auc_train_matrix))
auc.train.df <- data.frame(AUC=auc.train.df$V1,type="Train")
auc.test.df  <- as.data.frame(cbind(auc_test_matrix))
auc.test.df  <- data.frame(AUC=auc.test.df$V1,type="Test")
auc.lasso.df       <- rbind(auc.train.df, auc.test.df)
auc.lasso.df %>% group_by(type) %>%
  summarize(median_AUC = median(AUC))

lasso_AUCplots <- ggplot(auc.lasso.df) + geom_boxplot(aes(type, AUC, color = type)) +
  ggtitle("AUC scores for Train and Test \n (LASSO model)")
lasso_AUCplots
ggsave("Lasso.AUC.png", w = 6, h = 4, dpi = "screen")

# RIDGE
get_lambda_ridge <- function(r){
  # executes cv.glmnet fit, then applies lambdd.min() to get lambda.min
  lambda_min(cv.glmnet(r$X.train, r$y.train, family = "binomial",alpha = 0,
                       type.measure="auc", intercept = TRUE, parallel = TRUE))
}

get_ridge0         <- function(r, lam) {
  # fits ridge model and returns intercept
  ridge.a0_tag(glmnet(r$X.train, r$y.train,lambda = lam, alpha = 0,
                      family = "binomial",intercept = TRUE, parallel = TRUE, weights = r$wt))
}
get_betaHat.ridge        <- function(r, lam) {
  # fits ridge model and returns beta coefficients as a vector
  ridge.betaHat_tag(glmnet(r$X.train, r$y.train,lambda = lam, alpha = 0,
                           family = "binomial",intercept = TRUE, parallel = TRUE, weights = r$wt))
}


# register cores for parallel computing
registerDoMC(cores = 2)

system.time(lambda.mins_listed_ridge <- lapply(tt_reps, get_lambda_ridge))  
system.time(ridge0map <- pmap(list(tt_reps, lambda.mins_listed_ridge) , get_ridge0)) 
system.time(betamap_ridge   <- pmap(list(tt_reps, lambda.mins_listed_ridge) , get_betaHat.ridge)) 

for (r in seq(50)) {
  X.train = tt_reps[[r]]$X.train
  y.train = tt_reps[[r]]$y.train
  X.test  = tt_reps[[r]]$X.test
  y.test  = tt_reps[[r]]$y.test
  ridge0.hat     = ridge0map[[r]]
  ridge.beta.hat = betamap_ridge[[r]]
  
  for (i in c(1:thta.length)){
    prob.train     = exp(X.train %*% ridge.beta.hat + ridge0.hat)/(1 + exp(X.train %*% ridge.beta.hat +  ridge0.hat))
    prob.test      = exp(X.test %*% ridge.beta.hat + ridge0.hat)/(1 + exp(X.test %*% ridge.beta.hat +  ridge0.hat))
    P.train        = sum(y.train==1) # total positives in the data
    N.train        = sum(y.train==0) # total negatives in the data
    
    # calculate the FPR and TPR for train data 
    y.hat.train           = ifelse(prob.train > thta[i], 1, 0) # table(y.hat.train, y.train)
    FP.train              = sum(y.train[y.hat.train==1] == 0) # false positives 
    TP.train              = sum(y.hat.train[y.train==1] == 1) # true positives 
    FPR.train_matrix[i,r] = FP.train/N.train # FPR = type 1 error = 1 - specificity
    TPR.train_matrix[i,r] = TP.train/P.train # TPR = 1 - type 2 error = sensitivity
    
    # calculate the FPR and TPR for test data 
    y.hat.test           = ifelse(prob.test > thta[i], 1, 0)
    FP.test              = sum(y.test[y.hat.test==1] == 0) 
    TP.test              = sum(y.hat.test[y.test==1] == 1) 
    P.test               = sum(y.test==1) 
    N.test               = sum(y.test==0) 
    FPR.test_matrix[i,r] = FP.test/N.test 
    TPR.test_matrix[i,r] = TP.test/P.test
  }
}

#AUC RIDGE boxplots
auc_train_matrix_ridge <- matrix(0, Reps)
auc_test_matrix_ridge <- matrix(0, Reps)

for (i in c(1:Reps)) {
  auc_train_matrix_ridge[i]  <- sum((TPR.train_matrix[1:(thta.length-1),i] + 0.5 * diff(TPR.train_matrix[,i])) * diff(FPR.train_matrix[,i]))
  auc_test_matrix_ridge[i]   <- sum((TPR.test_matrix[1:(thta.length-1),i] + 0.5 * diff(TPR.test_matrix[,i])) * diff(FPR.test_matrix[,i]))
}


auc.train.df <- as.data.frame(cbind(auc_train_matrix_ridge))
auc.train.df <- data.frame(AUC=auc.train.df$V1,type="Train")
auc.test.df  <- as.data.frame(cbind(auc_test_matrix_ridge))
auc.test.df  <- data.frame(AUC=auc.test.df$V1,type="Test")
auc.ridge.df       <- rbind(auc.train.df, auc.test.df)

ggplot(auc.ridge.df) + geom_boxplot(aes(type, AUC, color = type)) +
  ggtitle("Comparison AUC scores for Train and Test \n (RIDGE model)")

#Median Test Ridge AUC
median(auc.test.df[,1])
#0.9926818                                                         

#Time to run single CV
system.time(lambda.mins_listed_ridge <- lapply(tt_reps[1], get_lambda)) 
#6.3 minutes

#Time to fit entire data set                                                          
Rprof()
cv.ridge.all = cv.glmnet(X, y, family = "binomial", alpha = 0, nfolds =10, type.measure="auc",
                         intercept = TRUE, weights = wt)
cv.ridge.fit.all <- glmnet(X, y, lambda = cv.ridge.all$lambda.min, 
                           family = "binomial", alpha = 0,  intercept = TRUE, weights = wt)
Rprof(NULL)    ## Turn off the profiler
prof_summary <- summaryRprof()
prof_summary$by.self
prof_summary$by.self[which.max(prof_summary$by.self$self.time),]
glue('Total time to fit single CV Ridge model: ',round({prof_summary$sampling.time / 60}, 2), ' minutes')
# Total time to fit single CV Ridge model: 7.85 minutes



# Comparing CV curves for LASSO, RIDGE and ELNET
# cv.glmnet fit on entire dataset
system.time(cv.lasso <- cv.glmnet(tt_reps[[1]]$X.train, tt_reps[[1]]$y.train, 
                                  family = "binomial", 
                                  alpha = 1,
                                  type.measure="auc",
                                  intercept = TRUE))
par(mfrow = c(1,2))        
plot(cv.lasso)
title("LASSO", line = 2.5)
points(c(log(cv.lasso$lambda.min), log(cv.lasso$lambda.1se)), c(max(cv.lasso$cvm),cv.lasso$cvm[which(cv.lasso$lambda == cv.lasso$lambda.1se)]),
       pch = 9, cex = 2, col = "blue")
text(log(cv.lasso$lambda.min), 0.97, TeX("$ Log(\\lambda)$ Min"), cex = 1.2,
     pos = 2)
log(cv.lasso$lambda.min)
log(cv.lasso$lambda.1se)
rocs <- roc.glmnet(cv.lasso$fit.preval, newy = y)
best <- cv.lasso$index["min",]
plot(rocs[[best]], type = "l")
invisible(sapply(rocs, lines, col = 'grey'))
lines(rocs[[best]], lwd = 3, col = "red")
title(main = TeX('ROC curve of optimal $\\lambda$',bold = TRUE))


system.time(cv.ridge <- cv.glmnet(tt_reps[[1]]$X.train, tt_reps[[1]]$y.train, family = "binomial", alpha = 0,type.measure="auc",
                                  intercept = TRUE, keep = TRUE, parallel = TRUE))
par(mfrow = c(3,2))


plot(cv.ridge)
title("RIDGE", line = 2.5)
points(c(log(cv.ridge$lambda.min), log(cv.ridge$lambda.1se)), 
       c(max(cv.ridge$cvm),cv.lasso$cvm[which(cv.ridge$lambda == cv.ridge$lambda.1se)]),
       pch = 9, cex = 2, col = "blue")
rocs <- roc.glmnet(cv.ridge$fit.preval, newy = y)
best <- cv.ridge$index["min",]
plot(rocs[[best]], type = "l")
invisible(sapply(rocs, lines, col = 'grey'))
lines(rocs[[best]], lwd = 3, col = "red")
title(main = TeX('ROC curve of optimal $\\lambda$',bold = TRUE))

system.time(cv.elnet <- cv.glmnet(tt_reps[[1]]$X.train, tt_reps[[1]]$y.train, family = "binomial", alpha = 0.5,type.measure="auc",
                                  intercept = TRUE, keep = TRUE, parallel = TRUE))
plot(cv.elnet)
title("Elastic Net", line = 2.5)
points(c(log(cv.elnet$lambda.min), log(cv.elnet$lambda.1se)),
       c(max(cv.elnet$cvm),cv.lasso$cvm[which(cv.elnet$lambda == cv.elnet$lambda.1se)]),
       pch = 9, cex = 2, col = "blue")
rocs <- roc.glmnet(cv.elnet$fit.preval, newy = y)
best <- cv.elnet$index["min",]
plot(rocs[[best]], type = "l")
invisible(sapply(rocs, lines, col = 'grey'))
lines(rocs[[best]], lwd = 3, col = "red")
title(main = TeX('ROC curve of optimal $\\lambda$',bold = TRUE))

# ELNET STANDARDIZED COEFFICIENTS
neg.wt.fullmodel = sum(y==1)/sum(y==0); 
wts.fullmodel = ifelse(y==0, neg.wt.fullmodel, 1)

elnet = cv.glmnet(X, y, family = "binomial", alpha = 0.5, nfolds = 10, type.measure="auc",
                  intercept=TRUE, weights = wts.fullmodel, standardize=TRUE)
elnet <- glmnet(X, y, lambda=elnet$lambda.min, family="binomial",
                alpha=0.5, intercept=TRUE, weights=wts.fullmodel, standardize=TRUE)

elnet.beta0.hat = elnet$a0
elnet.beta.hat = as.vector(elnet$beta)

obh = order(elnet.beta.hat, decreasing=T)
hbo = order(elnet.beta.hat, decreasing=F)

#EL-NET STANDARDIZED COEFFICIENTS BARPLOTS
elnet.decreasing.df = data.frame(features = colnames(X)[obh[1:15]],
                                 coefs = elnet.beta.hat[obh[1:15]])
elnet.decreasing.df$features <- factor(elnet.decreasing.df$features, levels = rev(colnames(X)[obh[1:15]]))
elnet.decreasing.bp <- ggplot(data=elnet.decreasing.df, aes(x=coefs, y=features)) +
  geom_bar(stat='identity', color='black', fill='lightskyblue')

elnet.increasing.df = data.frame(features = colnames(X)[hbo[1:15]],
                                 coefs = elnet.beta.hat[hbo[1:15]])
elnet.increasing.df$features <- factor(elnet.increasing.df$features, levels = rev(colnames(X)[hbo[1:15]]))
elnet.increasing.bp <- ggplot(data=elnet.increasing.df, aes(x=coefs, y=features)) +
  geom_bar(stat='identity', color='black', fill='orange')


# Feature importance plots
cv.lasso <- cv.glmnet(X, y,family = "binomial", 
                      alpha = 1,type.measure="auc",
                      intercept = TRUE, parallel = TRUE)
lasso_full <- glmnet(X, y, lambda=cv.lasso$lambda.min, family="binomial",
                     alpha=1, intercept=TRUE, weights=wts, parallel = TRUE)
lasso.betas.df <- data.frame(c(1:p), as.vector(lasso_full$beta))
lasso.features <- dimnames(X)[[2]]
colnames(lasso.betas.df) = c( "feature", "value")
lasso.betas.df$feature   = lasso.features
lasso.betas.df <- lasso.betas.df %>% mutate(pos = value >=0)
lasso.betas.df <- lasso.betas.df[obh,] 

# ordered based on ElNet model
lsPlot.spam.feat <-  ggplot(lasso.betas.df,aes(x= fct_rev(reorder(feature, -obh)),
                                               y=value, fill = pos)) +
  geom_col(position = "identity", colour="grey60", size = 0.1) + ylab('coefs') + 
  scale_x_discrete(name = "feature") + ggtitle("Lasso") +
  theme(plot.title = element_text(hjust = 0.5), axis.title.x=element_blank(),
        axis.text.x=element_blank(),axis.ticks.x=element_blank(),
        legend.position = "none")
lsPlot.spam.feat

elnet.betas.df = data.frame(c(1:p), as.vector(elnet$beta))
elnet.features = dimnames(X)[[2]]
colnames(elnet.betas.df) = c( "feature", "value")
elnet.betas.df$feature   = elnet.features
elnet.betas.df <- elnet.betas.df %>% mutate(pos = value >=0)

elnetPlot.spam.feat <-  ggplot(elnet.betas.df,aes(x= fct_rev(reorder(feature, value)),
                                               y=value, fill = pos)) +
  geom_col(position = "identity", colour="grey60", size = 0.1) + ylab('coefs') + 
  scale_x_discrete(name = "feature") + ggtitle("Elastic Net") +
  theme(plot.title = element_text(hjust = 0.5), axis.title.x=element_blank(),
        axis.text.x=element_blank(),axis.ticks.x=element_blank(),
        legend.position = "none")


ridge_full <- glmnet(X, y, lambda=cv.ridge$lambda.min, family="binomial",
                     alpha=0, intercept=TRUE, weights=wts, parallel = TRUE)
ridge.betas.df <- data.frame(c(1:p), as.vector(ridge_full$beta))
ridge.features <- dimnames(X)[[2]]
colnames(ridge.betas.df) <- c( "feature", "value")
ridge.betas.df$feature   <- ridge.features
ridge.betas.df <- ridge.betas.df %>% mutate(pos = value >=0)
ridge.betas.df <- ridge.betas.df[obh,]

ridgePlot.spam.feat <-  ggplot(ridge.betas.df,aes(x= fct_rev(reorder(feature, -obh)),
                                                  y=value, fill = pos)) +
  geom_col(position = "identity", colour="grey60", size = 0.1) + ylab('coefs') + 
  scale_x_discrete(name = "feature") + ggtitle("Ridge") +
  theme(plot.title = element_text(hjust = 0.5), axis.text.x=element_blank(),
        legend.position = "none")

# Feature importance plots
grid.arrange(elnetPlot.spam.feat,lsPlot.spam.feat, ridgePlot.spam.feat ,nrow = 3)


### ELNET 50 REPS
library(MESS)
auc.matrix.elnet <- matrix(0, nrow = 50, ncol = 3)

for (i in c(1:Reps)) {
  set.seed(i)
  idx = sample(n)
  train = idx[1:n.train]
  test = idx[(n.train+1):n]
  X.train = X[train,]
  X.test = X[test,]
  y.train = y[train]
  y.test = y[test]
  neg.wt = sum(y.train==1)/sum(y.train==0)
  wts = ifelse(y.train==0, neg.wt, 1)
  
  FPR.train.elnet = matrix(0, thta.length)
  TPR.train.elnet = matrix(0, thta.length)
  FPR.test.elnet = matrix(0, thta.length)
  TPR.test.elnet = matrix(0, thta.length)
  
  cv.elnet = cv.glmnet(X.train, y.train, family = "binomial", alpha = 0.5, nfolds = 10, type.measure="auc", intercept = TRUE, weights = wts, trace.it=1)
  elnet = glmnet(X.train, y.train, lambda=cv.elnet$lambda.min, family="binomial", alpha=0.5, intercept=TRUE, weights=wts)
  
  elnet.beta0.hat = elnet$a0
  elnet.beta.hat = as.vector(elnet$beta)
  
  for (j in c(1:thta.length)){
    prob.train = exp(X.train %*% elnet.beta.hat + elnet.beta0.hat)/(1 + exp(X.train %*% elnet.beta.hat + elnet.beta0.hat))
    prob.test = exp(X.test %*% elnet.beta.hat + elnet.beta0.hat)/(1 + exp(X.test %*% elnet.beta.hat + elnet.beta0.hat))
    P.train = sum(y.train==1) 
    N.train = sum(y.train==0)
    
    y.hat.train = ifelse(prob.train > thta[j], 1, 0)
    FP.train = sum(y.train[y.hat.train==1] == 0) 
    TP.train = sum(y.hat.train[y.train==1] == 1) 
    FPR.train.elnet[j]  = FP.train/N.train 
    TPR.train.elnet[j]  = TP.train/P.train
    
    y.hat.test = ifelse(prob.test > thta[j], 1, 0)
    FP.test = sum(y.test[y.hat.test==1] == 0) 
    TP.test = sum(y.hat.test[y.test==1] == 1) 
    P.test = sum(y.test==1) 
    N.test = sum(y.test==0) 
    FPR.test.elnet[j] = FP.test/N.test 
    TPR.test.elnet[j] = TP.test/P.test
  }
  
  auc.matrix.elnet[i,1] <- auc(FPR.train.elnet, TPR.train.elnet)
  auc.matrix.elnet[i,2] <- auc(FPR.test.elnet,TPR.test.elnet)
  auc.matrix.elnet[i,3] <- i
}

d.elnet <- data.frame(AUC = auc.matrix.elnet[,1], type = "TRAIN")
f.elnet <- data.frame(AUC = auc.matrix.elnet[,2], type = "TEST")
df.elnet <- rbind(d.elnet,f.elnet)
df.elnet$type <- factor(df.elnet$type, levels = c("TRAIN", "TEST"))

boxplot.theme = theme(axis.title.x = element_blank(),
                      axis.title.y = element_text(size=14),
                      axis.text.x = element_text(size=12),
                      axis.text.y = element_text(size=12),
                      legend.title = element_blank(),
                      plot.title = element_text(hjust=0.5, size=16))

boxplot.elnet <- df.elnet %>%
  ggplot(aes(x=type, y=AUC, fill=type)) +
  geom_boxplot() +
  ggtitle("Elastic-Net Train & Test AUCs") +
  theme_bw() +
  boxplot.theme

### RANDOM-FOREST BARPLOTS
rf.df = data.frame(x=X, y=as.factor(y))
rf = randomForest(y~., data=rf.df, mtry=sqrt(p), classwt=c(neg.wt.fullmodel,1), do.trace=T)

plot(rf)
varImpPlot(rf)
importance(rf)

rf.decreasing.df = data.frame(features=colnames(rf.df)[obh[1:15]],
                              MeanDecreaseGini=importance(rf)[obh[1:15]])

for (i in c(1:nrow(rf.decreasing.df))) {
  x = rf.decreasing.df$features[i]
  renamed = substring(x,3,nchar(x))
  rf.decreasing.df$features[i] <- renamed 
}

rf.decreasing.df <- rf.decreasing.df[order(rf.decreasing.df[,2], decreasing=T),]
rf.decreasing.df$features <- factor(rf.decreasing.df$features, levels = rev(rf.decreasing.df$features))
rf.decreasing.Barplot <- ggplot(data=rf.decreasing.df, aes(x=MeanDecreaseGini, y=features)) +
  geom_bar(stat='identity', color='black', fill='lightskyblue') +
  ggtitle("Random Forest") +
  theme(plot.title = element_text(hjust = 0.5))

rf.increasing.df = data.frame(features=colnames(rf.df)[hbo[1:15]],
                              MeanDecreaseGini=importance(rf)[hbo[1:15]])

for (i in c(1:nrow(rf.increasing.df))) {
  x = rf.increasing.df$features[i]
  renamed = substring(x,3,nchar(x))
  rf.increasing.df$features[i] <- renamed 
}

rf.increasing.df  <- rf.increasing.df[order(rf.increasing.df[,2], decreasing=T),]
rf.increasing.df$features <- factor(rf.increasing.df$features, levels = rev(rf.increasing.df$features))
rf.increasing.Barplot <- ggplot(data=rf.increasing.df, aes(x=MeanDecreaseGini, y=features)) +
  geom_bar(stat='identity', color='black', fill='orange') +
  ggtitle("Random Forest") +
  theme(plot.title = element_text(hjust = 0.5))

### RANDOM FOREST 50 LOOPS
library(ROCR) # for prediction(), performance() functions
auc.matrix.rf <- matrix(0, nrow=50, ncol=3)

for (i in c(1:Reps)) {
  set.seed(i)
  idx = sample(n)
  train = idx[1:n.train]
  test = idx[(n.train+1):n]
  X.train = X[train,]
  X.test = X[test,]
  y.train = y[train]
  y.test = y[test]
  
  rf.df.train = data.frame(x=X.train, y=as.factor(y.train))
  rf.train = randomForest(y~., data=rf.df.train, mtry=sqrt(p), classwt=c(neg.wt.fullmodel,1), do.trace=T)
  pred.train <- prediction(as.vector(rf.train$votes[,2]), rf.df.train$y)
  auc.train.rf <- performance(pred.train, measure="auc")@y.values[[1]] 
  
  rf.df.test = data.frame(x=X.test, y=as.factor(y.test))
  test.pred <- predict(rf.train, type="prob", newdata=rf.df.test)[,2]
  pred.test <- prediction(test.pred, rf.df.test$y)
  auc.test.rf <- performance(pred.test, measure="auc")@y.values[[1]]
  
  auc.matrix.rf[i,1] <- auc.train.rf
  auc.matrix.rf[i,2] <- auc.test.rf
  auc.matrix.rf[i,3] <- i
}

d.rf <- data.frame(AUC = auc.matrix.rf[,1], type = "TRAIN")
f.rf <- data.frame(AUC = auc.matrix.rf[,2], type = "TEST")
df.rf <- rbind(d.rf,f.rf)
df.rf$type <- factor(df.rf$type, levels = c("TRAIN", "TEST"))

rf.boxplot <- df.rf %>%
  ggplot(aes(x=type, y=AUC, fill=type)) +
  geom_boxplot() +
  ggtitle("Random Forest Train & Test AUCs") +
  theme_bw() +
  boxplot.theme