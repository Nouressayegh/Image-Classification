#%%
#On importe les bibliothèques dont nous avons besoin
import numpy as np
import matplotlib.pyplot as plt

#Load le fichier .mat
import scipy.io as sio

#Fonctions pour le Logistic Regression
from Fonctions import fGetShapeFeat,fClassify_LogisticReg,fTrain_LogisticReg

#Definir les training sets et testsing sets
from sklearn.model_selection import train_test_split
import random

#Evaluation du model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import statistics as st
from sklearn.metrics import roc_auc_score

#Other machine learning algorithms
from sklearn.linear_model import LogisticRegression
import sklearn.tree
import sklearn.neighbors
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


#%%
"""Les images à étudier et leur description"""

#load the images
H=sio.loadmat("Horizontal_edges.mat")

#load the elements
horiz_edges=H.get("horiz_edges")
images_croped=H.get("images_croped")
labels=H.get("labels")
names=H.get("names")

#show the un exemple of the images
plt.imshow(horiz_edges[0,1],cmap=plt.cm.gray)
plt.show()

# Number of images of cutting edges
p,num_edges = horiz_edges.shape;

# Number of features (in the case of ShapeFeat it is 10)
num_features = 10;

#Initialiation of matrix of descriptors. 
X = np.zeros((num_edges, num_features));

#Fill the X matrix with the shape descriptors we need.
for i in range(num_edges):
    
    edge = horiz_edges[0,i]
    
    desc_edge_i = fGetShapeFeat(edge);
    
    X[i,:] = desc_edge_i;

# Create the vector of labels Y. 0 low or medium wear and 1 for high wear level.
Y = labels[:,1]>=2;
for i in range(10):
    mu_data = X[:,i].mean()
    std_data = X[:,i].std()
    X[:,i] = (X[:,i]-mu_data)/std_data
#%%
""" Classification croisée """
n_samples = 100
accuracylist = []
Fscorelist = []
lr_auclist=[]
accuracy1 = np.zeros(n_samples)
accuracy2 = np.zeros(n_samples)
accuracy3 = np.zeros(n_samples)
accuracy4 = np.zeros(n_samples)
accuracy5 = np.zeros(n_samples)
accuracy6 = np.zeros(n_samples)
accuracy7 = np.zeros(n_samples)
accuracy8 = np.zeros(n_samples)

lr_auc1=np.zeros(n_samples)
lr_auc2=np.zeros(n_samples)
lr_auc3=np.zeros(n_samples)
lr_auc4=np.zeros(n_samples)
lr_auc5=np.zeros(n_samples)
lr_auc6=np.zeros(n_samples)
lr_auc7=np.zeros(n_samples)
lr_auc8=np.zeros(n_samples)

for k in range(n_samples):
    print('sample : ', k)
    #splitting data into training and test sets
    random.seed(k)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
    
    alpha = 0.1;

    theta ,J = fTrain_LogisticReg(X_train, Y_train, alpha);

    # CLASSIFICATION OF THE TEST SET
    Y_test_hat = fClassify_LogisticReg(X_test, theta);

    Y_test_pred=Y_test_hat>=0.5
    
    M=confusion_matrix(Y_test, Y_test_pred);
    accuracy=np.trace(M)/sum(sum(M))
    Precision=M[0,0]/(M[0,0]+M[1,0])
    Recall=M[0,0]/(M[0,0]+M[0,1])
    FScore=2*((Precision*Recall)/(Precision+Recall))
    
    accuracylist.append(accuracy)
    Fscorelist.append(FScore)
    lr_auclist.append( roc_auc_score(Y_test, Y_test_pred))
    
    logistic = LogisticRegression() #Logistic regression
    tree = sklearn.tree.DecisionTreeClassifier(max_depth=3) #Decision tree
    gradient_descent=SGDClassifier()# stochastic gradient descent
    gradient_boosting=GradientBoostingClassifier()# gradient boosting
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=13) #k nearest neighbors
    gnb= GaussianNB() #Naive bayesian 

    SVM_linear=svm.SVC(kernel='linear')
    
#Train models with the method .fit()
    model1 = logistic.fit(X_train, Y_train)

    model2 = tree.fit(X_train, Y_train)

    model3 = gradient_descent.fit(X_train, Y_train)

    model4 = gradient_boosting.fit(X_train, Y_train)

    model5 = knn.fit(X_train, Y_train)

    model6 = gnb.fit(X_train, Y_train)
    
    model7=SVM_linear.fit(X_train,Y_train)

    predictions1 = logistic.predict(X_test)
    predictions2 = tree.predict(X_test)
    predictions3 = gradient_descent.predict(X_test)
    predictions4 = gradient_boosting.predict(X_test)
    predictions5 = knn.predict(X_test)
    predictions6 = gnb.predict(X_test)
    predictions7= SVM_linear.predict(X_test)
    
    lr_auc1[k] = roc_auc_score(Y_test,predictions1)
    lr_auc2[k] = roc_auc_score(Y_test,predictions2)
    lr_auc3[k] = roc_auc_score(Y_test, predictions3)
    lr_auc4[k] = roc_auc_score(Y_test, predictions4)
    lr_auc5[k] = roc_auc_score(Y_test, predictions5)
    lr_auc6[k] = roc_auc_score(Y_test, predictions6)
    lr_auc7[k] = roc_auc_score(Y_test, predictions7)
    
    accuracy1[k] = accuracy_score(Y_test,predictions1)
    accuracy2[k] = accuracy_score(Y_test,predictions2)
    accuracy3[k] = accuracy_score(Y_test,predictions3)
    accuracy4[k] = accuracy_score(Y_test,predictions4)
    accuracy5[k] = accuracy_score(Y_test,predictions5)
    accuracy6[k] = accuracy_score(Y_test,predictions6)
    accuracy7[k] = accuracy_score(Y_test,predictions7)

accuracy_moyenne=st.mean(accuracylist)
Fscore_moyenne=st.mean(Fscorelist)
lr_auc_moyenne = st.mean(lr_auclist)


lr_auc1_m = st.mean(lr_auc1)

lr_auc2_m = st.mean(lr_auc2)

lr_auc3_m = st.mean(lr_auc3)

lr_auc4_m = st.mean(lr_auc4)

lr_auc5_m = st.mean(lr_auc5)

lr_auc6_m = st.mean(lr_auc6)

lr_auc7_m = st.mean(lr_auc7)

accuracy1_m = st.mean(accuracy1)
accuracy2_m = st.mean(accuracy2)
accuracy3_m = st.mean(accuracy3)
accuracy4_m = st.mean(accuracy4)
accuracy5_m = st.mean(accuracy5)
accuracy6_m = st.mean(accuracy6)
accuracy7_m = st.mean(accuracy7)

fig=plt.figure()
plt.bar([1,2,3,4,5,6,7,8],height=[accuracy_moyenne,accuracy1_m,accuracy2_m,accuracy3_m,accuracy4_m,accuracy5_m,accuracy6_m,accuracy7_m],tick_label=['model','Logitic \n regression', 'Decision \n tree', 'Gradient \n descent', 'Gradient \n boosting', 'Knn','Naive \n bayesian','svm'])
plt.title('models accuracy')
plt.ylabel('accuracy')
plt.xlabel('model')
plt.show()

fig=plt.figure()
plt.bar([1,2,3,4,5,6,7,8],height=[lr_auc_moyenne,lr_auc1_m,lr_auc2_m,lr_auc3_m,lr_auc4_m,lr_auc5_m,lr_auc6_m,lr_auc7_m],tick_label=['model','Logitic \n regression', 'Decision \n tree', 'Gradient \n descent', 'Gradient \n boosting', 'Knn','Naive\n bayesian','svm'])
plt.title('models AUC')
plt.ylabel('AUC')
plt.xlabel('model')
plt.show()
