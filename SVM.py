from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import random
from sklearn.metrics import accuracy_score
from Fonctions import fun_sigmoid,fGetShapeFeat,fCalculateCostLogReg,fClassify_LogisticReg,fTrain_LogisticReg
from sklearn.model_selection import train_test_split
import scipy.io as sio

H=sio.loadmat("Horizontal_edges.mat")

#load the elements
horiz_edges=H.get("horiz_edges")
images_croped=H.get("images_croped")
labels=H.get("labels")
names=H.get("names")

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
    
    
accuracy0 =[]
accuracy1 =[]
accuracy2 =[]
accuracy3 =[]
accuracy4 =[]


lr_auc0=[]
lr_auc1=[]
lr_auc2=[]
lr_auc3=[]
lr_auc4=[]


n_samples=100
for k in range(n_samples):
    print('sample : ', k)
    #splitting data into training and test sets
    random.seed(k)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
    
    alpha = 0.1;

    theta ,J = fTrain_LogisticReg(X_train, Y_train, alpha);

    # CLASSIFICATION OF THE TEST SET
    Y_test_hat = fClassify_LogisticReg(X_test, theta);

    Y_test_pred0=Y_test_hat>=0.5
    
    #linear svm
    linear=svm.SVC(kernel='linear',C=10)
    linear.fit(X_train,Y_train)
    Y_pred1=linear.predict(X_test)
    
    #polynomial svm
    poly=svm.SVC(kernel='poly',C=10)
    poly.fit(X_train,Y_train)
    Y_pred2=poly.predict(X_test)

    #sigmaid svm
    sigmoid=svm.SVC(kernel='sigmoid',C=10)
    sigmoid.fit(X_train,Y_train)
    Y_pred3=sigmoid.predict(X_test)
    
    #rbf svm
    rbf=svm.SVC(kernel='rbf',C=10)
    rbf.fit(X_train,Y_train)
    Y_pred4=rbf.predict(X_test)
    
    accuracy0.append(accuracy_score(Y_test,Y_test_pred0))
    accuracy1.append(accuracy_score(Y_test,Y_pred1))
    accuracy2.append(accuracy_score(Y_test,Y_pred2))
    accuracy3.append(accuracy_score(Y_test,Y_pred3))
    accuracy4.append(accuracy_score(Y_test,Y_pred4))

    
    lr_auc0.append(roc_auc_score(Y_test, Y_test_pred0))
    lr_auc1.append(roc_auc_score(Y_test,Y_pred1))
    lr_auc2.append(roc_auc_score(Y_test,Y_pred2))
    lr_auc3.append(roc_auc_score(Y_test, Y_pred3))
    lr_auc4.append(roc_auc_score(Y_test, Y_pred4))

    
    
lr_auc0_m = st.mean(lr_auc0)
lr_auc1_m = st.mean(lr_auc1)
lr_auc2_m = st.mean(lr_auc2)
lr_auc3_m = st.mean(lr_auc3)
lr_auc4_m = st.mean(lr_auc4)




accuracy0_m = st.mean(accuracy0)
accuracy1_m = st.mean(accuracy1)
accuracy2_m = st.mean(accuracy2)
accuracy3_m = st.mean(accuracy3)
accuracy4_m = st.mean(accuracy4)



fig=plt.figure()
plt.bar([1,2,3,4,5],height=[accuracy0_m,accuracy1_m,accuracy2_m,accuracy3_m,accuracy4_m],tick_label=['model','Linear', 'poly', 'sigmoid', 'rbf'])
plt.title('models accuracy with soft margin')
plt.ylabel('accuracy')
plt.xlabel('model')
plt.show()

fig=plt.figure()
plt.bar([1,2,3,4,5],height=[lr_auc0_m,lr_auc1_m,lr_auc2_m,lr_auc3_m,lr_auc4_m],tick_label=['model','Linear', 'poly', 'sigmoid', 'rbf'])
plt.title('models AUC with soft margin')
plt.ylabel('AUC')
plt.xlabel('model')
plt.show()
