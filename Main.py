import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Fonctions import fGetShapeFeat,fClassify_LogisticReg,fTrain_LogisticReg,ROC

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


for i in range(num_edges):
    
    edge = horiz_edges[0,i]
    #the descrption of each edge
    desc_edge_i = fGetShapeFeat(edge);
    # Store the feature vector into the matrix X.
    X[i,:] = desc_edge_i;

# Create the vector of labels Y. 0 low or medium wear and 1 for high wear level.
Y = labels[:,1]>=2;

#%%
"""CLASSIFICATION""" 

num_patterns, num_features = X.shape;

# Normalization of the data
for i in range(10):
    mu_data = X[:,i].mean()
    std_data = X[:,i].std()
    X[:,i] = (X[:,i]-mu_data)/std_data
    
p_train = 0.6;

#SPLIT DATA INTO TRAINING AND TEST SETS

num_patterns_train = round(p_train*num_patterns);

indx_permutation = np.random.permutation(num_patterns);

indxs_train = indx_permutation[0:num_patterns_train];
indxs_test = indx_permutation[num_patterns_train:-1];

X_train = X[indxs_train, :];
Y_train = Y[indxs_train];

X_test= X[indxs_test, :];
Y_test = Y[indxs_test];


#%%
"""TRAINING OF THE CLASSIFIER AND CLASSIFICATION OF THE TEST SET"""
#Learning rate
alpha =2;
# TRAINING
theta ,J = fTrain_LogisticReg(X_train, Y_train, alpha);
# CLASSIFICATION OF THE TEST SET
Y_test_hat = fClassify_LogisticReg(X_test, theta);
Y_test_pred=Y_test_hat>=0.5

#Confusion matrix
M=confusion_matrix(Y_test, Y_test_pred);
print(M)

# ACCURACY AND F-SCORE

accuracy=np.trace(M)/sum(sum(M))
Precision=M[0,0]/(M[0,0]+M[1,0])
Recall=M[0,0]/(M[0,0]+M[0,1])
FScore=2*((Precision*Recall)/(Precision+Recall))

print('the accuracy :', accuracy*100);
print('the Fscore :', FScore);
#%%
#the ROC curve and the area under the curve

T,F=ROC(Y_test,Y_test_hat)
L=np.arange(0,1,0.001)
plt.plot(F,T)
plt.plot(L,L)
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#now we can the area under the curve using the rectangle rule
m=len(L)
AUC=0
for i in range(m-1):
    AUC+=(F[i]-F[i+1])*T[i]
print("The AUC is",AUC)

