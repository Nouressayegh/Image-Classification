import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from Function import fGetShapeFeat, lbp,AUC
import statistics as st

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import random
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import ShuffleSplit


from sklearn.metrics import f1_score,accuracy_score

#%%
labels = pd.read_csv ('ISIC-2017_Data_GroundTruth_Classification.csv')

Image=[]
redImage=[]
segmentation=[]
superpixels=[]
names=[]
Y=[]
print("Import the Data")
for i in range(519):
    if i<10 :
        name = r'C:\Users\noure\Desktop\TB3\projet2\PROJECT_Data\PROJECT_Data\ISIC_000000' + str(i)
        I = cv2.imread(name + '.jpg')
        Binaire = cv2.imread(name + '_segmentation.png', cv2.IMREAD_GRAYSCALE)
        Super = cv2.imread(name + '_superpixels.png')
        N = 'ISIC_000000' + str(i)
    if i>=10 and i<100:
        name = r'C:\Users\noure\Desktop\TB3\projet2\PROJECT_Data\PROJECT_Data\ISIC_00000' + str(i)
        I = cv2.imread(name + '.jpg')
        Binaire = cv2.imread(name + '_segmentation.png', cv2.IMREAD_GRAYSCALE)
        Super = cv2.imread(name + '_superpixels.png')
        N = 'ISIC_00000' + str(i)
    if i>=100:
        name = r'C:\Users\noure\Desktop\TB3\projet2\PROJECT_Data\PROJECT_Data\ISIC_0000' + str(i)
        I = cv2.imread(name + '.jpg')
        Binaire = cv2.imread(name + '_segmentation.png', cv2.IMREAD_GRAYSCALE)
        Super = cv2.imread(name + '_superpixels.png')
        N = 'ISIC_0000' + str(i)
        
    if I is None:
        #l'indice ne correspond pas à l'image
        j=False
    else: j=True
    
    if j:
        #Original images
        
        Image.append(I)
        #red composent of the image        
        Ired= cv2.resize(I[:,:,2], (767, 1022)) 
        redImage.append(Ired)
        #segmented image
        Binaire= cv2.resize(Binaire, (767, 1022)) 
        Binaire=Binaire>100
        Binaire=Binaire*1
        segmentation.append(Binaire)
        #superpixels image
        superpixels.append(Super)
        names.append(N)
        
    
    print(i)
#extraction of the labels
print("Extract the Labels")
for k in range (2000):
    if labels['image_id'][k] in names :
        print(k)
        Y.append(int(labels['melanoma'][k]))

#%%
#feature detection
print("The feature from the segmented image")
X_bin=np.zeros((200,9))
for i in range((len(segmentation))):
    I=segmentation[i]
    X_bin[i,:]=fGetShapeFeat(I)
    print(i)

#%%
#texture
#we will use the red component of all the image.
print("lbp")
LBP=[]
for i in range(200):
    LBP.append(lbp(redImage[i]))
    print(i)

#%%
X_texture=np.zeros((200,255))
for i in range (200):
    X_texture[i,:]=LBP[i]
#%%
#Superpixels processing
#Compute the number of superpixels in every image
print("Superpixels")
numbre_sp = []
for i in range(200):
    print(i)
    sp = superpixels[i]
    #The red component
    R = np.array(sp[:,:,2])
    R = np.reshape(R, (1,-1))
    R = R[0]
    
    #the green component
    G = np.array(sp[:,:,1])
    G = np.reshape(G, (1,-1))
    G = G[0]

    #indexes of the last layer which are significative
    indexes = [i for i in range(len(G)) if G[i] == 3]

    #extraction of the superpixels in this layer
    R_bis = R[indexes]
    
    #Removing repeated values
    R_sub = list(dict.fromkeys(R[indexes]))
    
    numbre_sp = numbre_sp + [max(R_sub)]

numbre_sp = np.array(numbre_sp)
numbre_sp = np.reshape(numbre_sp, (200,))

#%%
X=np.zeros((200,265))
X[:,0:9]=X_bin
X[:,9:264]=X_texture
X[:,264]=numbre_sp
#select the 200 best feautures
data = SelectKBest(chi2, k=200).fit_transform(X,Y)
for i in range(200):
    mu_data = data[:,i].mean()
    std_data = data[:,i].std()
    data[:,i] = (data[:,i]-mu_data)/std_data


#%%
n_samples = 100
classifier = ["KNN", "SVM L", "RBF", "Gradient \n descent",
         "Decision \n Tree", "Random \n Forest", 'Gradien \n Boosting',"Logistic \n gradient"]

accuracy1 =[] 
accuracy2 = []
accuracy3 = []
accuracy4 = []
accuracy5 = []
accuracy6 = []
accuracy7 = []
accuracy8 = []


Fscore1 =[] 
Fscore2 = []
Fscore3 = []
Fscore4 = []
Fscore5 = []
Fscore6 = []
Fscore7 = []
Fscore8 = []


lr_auc1=[]
lr_auc2=[]
lr_auc3=[]
lr_auc4=[]
lr_auc5=[]
lr_auc6=[]
lr_auc7=[]
lr_auc8=[]


for k in range(n_samples):
    print('sample : ', k)
    #splitting data into training and test sets
    random.seed(k)
    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size = 0.25)
    
    
    #k nearest neighbors
    knn = KNeighborsClassifier(n_neighbors=6,n_jobs=-1,leaf_size=1,algorithm='auto') 
    #linear svm
    svmL=svm.SVC(kernel="rbf",C=100,gamma=0.01)
    #RBF svm
    svmR=svm.SVC(kernel='linear',C=0.1,gamma=1)
    # stochastic gradient descent
    gradient_descent=SGDClassifier(alpha=0.01,l1_ratio=0.95,loss='hinge',penalty='elasticnet')
    #Decision tree
    tree = DecisionTreeClassifier(max_depth=5,min_samples_leaf=1) 
    #Random Forest
    forest=RandomForestClassifier(bootstrap=True,max_depth=5,max_features='sqrt',min_samples_leaf=2,min_samples_split=3,n_estimators=17)
    #gradient boosting
    grad_boost=GradientBoostingClassifier(learning_rate=0.15,max_features='auto',n_estimators=750,max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1, random_state=10)
    #logistic gradient
    logistic= LogisticRegression()

    model1 = knn.fit(X_train, Y_train)
    model2 = svmL.fit(X_train, Y_train)
    model3 = svmR.fit(X_train, Y_train)
    model4 = gradient_descent.fit(X_train, Y_train)
    model5 = tree.fit(X_train, Y_train)
    model6 = forest.fit(X_train, Y_train)
    model7 = grad_boost.fit(X_train, Y_train)
    model8 = logistic.fit(X_train, Y_train)

    predictions1 = knn.predict(X_test)
    predictions2 = svmL.predict(X_test)
    predictions3 = svmR.predict(X_test)
    predictions4 = gradient_descent.predict(X_test)
    predictions5 = tree.predict(X_test)
    predictions6 = forest.predict(X_test)
    predictions7 = grad_boost.predict(X_test)
    predictions8 = logistic.predict(X_test)

    accuracy1.append(accuracy_score(Y_test,predictions1))
    accuracy2.append(accuracy_score(Y_test,predictions2))
    accuracy3.append(accuracy_score(Y_test,predictions3))
    accuracy4.append( accuracy_score(Y_test,predictions4))
    accuracy5.append(accuracy_score(Y_test,predictions5))
    accuracy6.append( accuracy_score(Y_test,predictions6))
    accuracy7.append( accuracy_score(Y_test,predictions7))
    accuracy8.append( accuracy_score(Y_test,predictions8))
    
    Fscore1.append(f1_score(Y_test,predictions1))
    Fscore2.append( f1_score(Y_test,predictions2))
    Fscore3.append( f1_score(Y_test,predictions3))
    Fscore4.append( f1_score(Y_test,predictions4))
    Fscore5.append(f1_score(Y_test,predictions5))
    Fscore6.append(f1_score(Y_test,predictions6))
    Fscore7.append(f1_score(Y_test,predictions7))
    Fscore8.append(f1_score(Y_test,predictions8))

    lr_auc1.append(AUC(Y_test,predictions1))
    lr_auc2.append(AUC(Y_test,predictions2))
    lr_auc3.append(AUC(Y_test, predictions3))
    lr_auc4.append(AUC(Y_test, predictions4))
    lr_auc5.append(AUC(Y_test, predictions5))
    lr_auc6.append(AUC(Y_test, predictions6))
    lr_auc7.append(AUC(Y_test, predictions7))
    lr_auc8.append(AUC(Y_test, predictions8))


accuracy1_m =round( st.mean(accuracy1),2)
accuracy2_m =round( st.mean(accuracy2),2)
accuracy3_m =round( st.mean(accuracy3),2)
accuracy4_m = round( st.mean(accuracy4),2)
accuracy5_m = round( st.mean(accuracy5),2)
accuracy6_m =round( st.mean(accuracy6),2)
accuracy7_m =round( st.mean(accuracy7),2)
accuracy8_m =round( st.mean(accuracy8),2)

Fscore1_m=st.mean(Fscore1)
Fscore2_m=st.mean(Fscore2)
Fscore3_m=st.mean(Fscore3)
Fscore4_m=st.mean(Fscore4)
Fscore5_m=st.mean(Fscore5)
Fscore6_m=st.mean(Fscore6)
Fscore7_m=st.mean(Fscore7)
Fscore8_m=st.mean(Fscore8)

lr_auc1_m =round( st.mean(lr_auc1),2)
lr_auc2_m =round( st.mean(lr_auc2),2)
lr_auc3_m =round( st.mean(lr_auc3),2)
lr_auc4_m =round( st.mean(lr_auc4),2)
lr_auc5_m = round( st.mean(lr_auc5),2)
lr_auc6_m = round( st.mean(lr_auc6),2)
lr_auc7_m = round( st.mean(lr_auc7),2)
lr_auc8_m = round( st.mean(lr_auc8),2)

acc=[accuracy1_m,accuracy2_m,accuracy3_m,accuracy4_m,accuracy5_m,accuracy6_m,accuracy7_m,accuracy8_m]
auc=[lr_auc1_m,lr_auc2_m,lr_auc3_m,lr_auc4_m,lr_auc5_m,lr_auc6_m,lr_auc7_m,lr_auc8_m]
labels =classifier


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, acc, width, label='accuracy')
rects2 = ax.bar(x + width/2, auc, width, label='AUC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Desctiption of all the classifier')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


fig=plt.figure()
plt.bar([1,2,3,4,5,6,7,8],height=[Fscore1_m,Fscore2_m,Fscore3_m,Fscore4_m,Fscore5_m,Fscore6_m,Fscore7_m,Fscore8_m],tick_label=classifier)
plt.title('models Fscore')
plt.ylabel('Fscore')
plt.xlabel('model')
plt.show()

