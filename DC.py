import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_excel("preprocess.xlsx")
x=df[["a","b","c","d","e","f","g","h","i","j"]]
y=df["z"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(xtrain.values,ytrain)
ypred=model.predict(xtest.values)
df2=pd.DataFrame({"ytest":ytest,"ypred":ypred})
print(df2)
print("=============================")
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(ytest,ypred))
cm=confusion_matrix(ytest,ypred)
TP=cm[0,0]
TN=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]
specifity=TN/(TN+FP)
print("specifity=",(specifity))
print("=============================")
print(classification_report(ytest,ypred))
from sklearn.metrics import r2_score,accuracy_score
print("R2=",r2_score(ytest,ypred))
print("AC=",accuracy_score(ytest,ypred))
print("=============================")
#Overfitting
print("training set score=",model.score(xtrain.values,ytrain))
print("Test set score=",model.score(xtest.values,ytest))
print("=============================")
ypred1=model.predict([[1,0,0,0,0,0.25,1,0.9,0.25,1]])
print("PredDC=",ypred1)
print("=============================")
from sklearn import metrics
import numpy as np
#ROC AUC
from sklearn.metrics import roc_auc_score
ROC_AUC=roc_auc_score(ytest,ypred)
print("ROC_AUC=",(ROC_AUC))
#K-Fold Validation
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,xtrain,ytrain,cv=3,scoring="accuracy")
print("average 10fold-cross validation scores=",scores.mean())
print(scores)
#Plot Tree
from sklearn.tree import plot_tree
plot_tree(model)
plt.show()






