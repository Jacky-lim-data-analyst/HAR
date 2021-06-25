# Human activity recognition dataset can be obtained from 
# http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
# Reference: Jorge-L. Reyes-Ortiz, Luca Oneto, Albert Samà, Xavier Parra, Davide Anguita. Transition-Aware Human Activity Recognition Using Smartphones. Neurocomputing. Springer 2015.

# train dataset and its corresponding labels
data=read.table("Train/X_train.txt",header = FALSE, sep = " ", dec = ".")
labels=read.table("Train/y_train.txt",header=FALSE)

# test dataset and its corresponding labels
data_test=read.table("Test/X_test.txt",header=FALSE, sep= " ",dec = ".")
labels_test=read.table("Test/y_test.txt",header = FALSE)


# Insert header name ------------------------------------------------------

# features' names can be found in features.txt
featurenames=read.table("features.txt",header = FALSE)
# some of the feature names are redundant
featurenames=make.unique(featurenames[,1],sep="_")
colnames(data)=featurenames
colnames(data_test)=featurenames
# activity types are numeric
colnames(labels)="activity_types"
colnames(labels_test)="activity_types"


# map labels to its real behaviors ----------------------------------------

# activity labels are characters
act_labels=read.table("activity_labels.txt",header = FALSE)[,2]

library(plyr)
labels$at=mapvalues(labels$activity_types,from=c(1:12),to = act_labels)
labels_test$at=mapvalues(labels_test$activity_types,from=c(1:12),to=act_labels)

# Construct complete dataframe
X_train=cbind(data,labels)
# Filter the data to have activity types from 1 to 6.
X_train=X_train[(X_train$activity_types>=1) & (X_train$activity_types<=6),]
X_test=cbind(data_test,labels_test)
X_test=X_test[(X_test$activity_types>=1) & (X_test$activity_types<=6),]

# remove unwanted variables
rm(data,data_test,labels,labels_test)

# LDA classifier---------------------------------------------------------------------

library(MASS)
start_time=Sys.time()
# LDA model training
model_lda=lda(at~.-activity_types,data=X_train)
end_time=Sys.time()
(t_train_lda=end_time-start_time)
# Encounter the problems of collinearity
pred=predict(model_lda,newdata = X_train)$class
nclass=max(X_train$activity_types)

# performance metrics for multiclass problem
perform_metric=function(true_label,pred,nclass) {
  conf=table(truth=true_label,prediction=pred)    # confusion matrix
  nsample=sum(conf)
  acc=sum(diag(conf))/nsample   # accuracy
  Rowsum=rowSums(conf)
  Colsum=colSums(conf)
  # Kappa statistics
  expected_acc=sum(Rowsum*Colsum/nsample)/nsample
  Kappa=(acc-expected_acc)/(1-expected_acc)
  sensitivity=diag(conf)/Rowsum
  precision=diag(conf)/Colsum
  specificity=c()
  for (i in 1:nclass) {
    specificity[i]=sum(conf[-i,-i])/(sum(conf[-i,-i])+colSums(conf)[i]-diag(conf)[i])
  }
  names(specificity)=names(sensitivity)
  list(confusion_matrix=conf,
       accuracy=round(acc,4),Kappa_statistic=round(Kappa,4),
       sensitivity=round(sensitivity,4),specificity=round(specificity,4),
       precision=round(precision,4))
}

# Training performance
perform_metric(X_train$at,pred,nclass)
# test performance
pred=predict(model_lda,newdata = X_test)$class
perform_metric(X_test$at,pred,nclass)

# qda ---------------------------------------------------------------------

# model_qda=qda(at~.-activity_types,data=X_train)
# Error in qda.default(x, grouping, ...) : rank deficiency in group LAYING

# kNN ---------------------------------------------------------------------

library(class)
# set.seed(100)
kfold=5    # k-fold cv, k=5
nn=30     # maximum number of nearest neighbors

y_train=factor(X_train$at)
nfeatures=561    # total number of features

# function for searching the best k for kNN
kfold_knn_acc=function(data,label,kfold,nn,nfeatures,seed=100) {
  set.seed(seed)
  idx_kfold=sample(1:kfold,size=nrow(data),replace=TRUE)
  # Matrix to store the cv accuracy
  acc=matrix(,nrow=5,ncol=(nn-1))
  for (i in 1:kfold) {
    data_train=data[,1:nfeatures][idx_kfold!=i,]
    data_val=data[,1:nfeatures][idx_kfold==i,]
    labels_train=label[idx_kfold!=i]
    labels_val=label[idx_kfold==i]
    for (j in 2:nn) {
      pred_knn=knn(data_train,data_val,labels_train,k=j)
      acc[i,j-1]=mean(pred_knn==labels_val)
    } 
  }
  acc
}
start_time=Sys.time()
acc=kfold_knn_acc(X_train,y_train,kfold,nn,nfeatures)
end_time=Sys.time()
(t_cv_knn=end_time-start_time)
# around 40-45 minutes is needed. Save it as "cv_acc_knn.RData"
save(acc,file="cv_acc_knn_1.RData")
# load('cv_acc_knn.RData')

# Visualize the average accuracies for each choices of k
mean_acc=apply(acc,2,mean)
std_acc=apply(acc, 2, sd)
plot_acc_knn=data.frame(nn=c(2:30),mean_acc,std_acc)
k_opt= which.max(mean_acc)+1

library(ggplot2)
ggplot(data=plot_acc_knn,aes(x=nn,y=mean_acc,ymin=mean_acc-std_acc,ymax=mean_acc+std_acc)) +
  geom_line() + geom_point() + geom_errorbar() +
  geom_vline(xintercept = k_opt,color="red",linetype=2) +
  xlab("number of nearest neighbors") +
  ylab("accuracy")

# training performance
pred_knn=knn(X_train[,1:nfeatures],X_train[,1:nfeatures],y_train,k=k_opt)
perform_metric(y_train,pred_knn,nclass)

# test performance
y_test=factor(X_test$at)
pred_knn=knn(X_train[,1:nfeatures],X_test[,1:nfeatures],y_train,k=k_opt)
perform_metric(y_test,pred_knn,nclass)

# pca dimensionality reduction --------------------------------------------

pca=prcomp(X_train[,1:nfeatures])

# variance explained
var_explained=pca$sdev^2/sum(pca$sdev^2)
plot_var_exp=data.frame(no_prin_comp=c(1:10),var=var_explained[1:10])

# Scree plot
ggplot(data=plot_var_exp,aes(x=no_prin_comp,y=var)) +
  geom_line() + geom_point() +
  xlab("Number of principal components") +
  ylab("Variance explained")+
  ggtitle("Scree Plot")

# Visualize in grouped scatterplot
pca_plot=data.frame(pca$x[,c(1,2)],class=X_train$at)
ggplot(data=pca_plot,aes(x=PC1,y=PC2)) +
  geom_point(aes(color=factor(class)))

# cumulative percentages of variance explained
cum_var=cumsum(var_explained)
nf=length(cum_var[cum_var<=0.95])
X_pca=data.frame(pca$x[,1:(nf+1)])
X_pca=cbind(X_pca,at=X_train$at)

# Prepare test data
X_pca_test=predict(pca,newdata=X_test[,1:nfeatures])[,1:(nf+1)]
xx=as.data.frame(X_pca_test)
X_pca_test=cbind(xx,at=X_test$at)


# PCA + LDA ---------------------------------------------------------------

start_time=Sys.time()
model_lda=lda(at~.,data=X_pca)
end_time=Sys.time()
(t_train_pca_lda=end_time-start_time)

pred=predict(model_lda,newdata = X_pca)$class
perform_metric(X_pca$at,pred,nclass)  # Training
pred=predict(model_lda,newdata = X_pca_test)$class
perform_metric(X_pca_test$at,pred,nclass)   # test


# PCA+QDA -----------------------------------------------------------------

start_time=Sys.time()
model_qda=qda(at~.,data=X_pca)
end_time=Sys.time()
(t_train_pca_qda=end_time-start_time)

pred=predict(model_qda,newdata = X_pca)$class
perform_metric(X_pca$at,pred,nclass)   # Training
pred=predict(model_qda,newdata = X_pca_test)$class
perform_metric(X_pca_test$at,pred,nclass)   #test

# PCA + kNN ---------------------------------------------------------------

#set.seed(100)
kfold=5    # 5-fold cv
nn=30     # maximum number of nearest neighbors
y_train=factor(X_train$at)

# cv to get the best k
start_time=Sys.time()
acc_mat=kfold_knn_acc(X_pca,y_train,kfold,nn,nf+1)
end_time=Sys.time()
(t_train_pca_knn=end_time-start_time)

mean_acc=apply(acc_mat,2,mean)
std_acc=apply(acc_mat, 2, sd)
plot_acc_knn=data.frame(nn=c(2:nn),mean_acc,std_acc)
k_opt=which.max(mean_acc)+1

ggplot(data=plot_acc_knn,aes(x=nn,y=mean_acc,ymin=mean_acc-std_acc,ymax=mean_acc+std_acc)) +
  geom_line() + geom_point() + geom_errorbar() +
  geom_vline(xintercept = k_opt,color="red",linetype=2) +
  xlab("number of nearest neighbors") +
  ylab("accuracy")
# k=3 is optimal

# Training performance
pred_knn=knn(X_pca[,1:nf+1],X_pca[,1:nf+1],y_train,k=k_opt)
perform_metric(y_train,pred_knn,nclass)
# Test performance
y_test=factor(X_pca_test$at)
pred_knn=knn(X_pca[,1:nf+1],X_pca_test[,1:nf+1],y_train,k=k_opt)
perform_metric(y_test,pred_knn,nclass)

# original paper results --------------------------------------------------

# ESANN 2013 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence
# and Machine Learning. Bruges (Belgium), 24-26 April 2013
cmat=cbind(c(492,18,4,0,0,0),c(1,451,6,2,0,0),c(3,2,410,0,0,0),
           c(0,0,0,432,14,0),c(0,0,0,57,518,0),c(0,0,0,0,0,537))
colnames(cmat)=act_labels[1:6]
rownames(cmat)=act_labels[1:6]
nsample=sum(cmat)
(acc=sum(diag(cmat))/nsample)  # accuracy
Rowsum=rowSums(cmat)
Colsum=colSums(cmat)
# Kappa statistics
expected_acc=sum(Rowsum*Colsum/nsample)/nsample
((acc-expected_acc)/(1-expected_acc))
(sensitivity=diag(cmat)/Rowsum)
(precision=diag(cmat)/Colsum)
