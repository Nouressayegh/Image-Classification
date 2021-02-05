
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt


def fGetShapeFeat(region):

    # Initialise the output
    shapeFeatVector =np.zeros(10);
    Feat=sm.regionprops(region)

    # Convex Area: 
    shapeFeatVector[0] = Feat[0].area

    # Eccentricity: 
    shapeFeatVector[1] = Feat[0].eccentricity

    # Perimeter: 
    shapeFeatVector[2] = Feat[0].perimeter

    #Equivalent Diameter: 
    shapeFeatVector[3] = Feat[0].equivalent_diameter

    # Extent: 
    shapeFeatVector[4] =Feat[0].extent

    # Filled Area: 
    shapeFeatVector[5] = Feat[0].filled_area

    # Minor Axis Length: 
    shapeFeatVector[6] = Feat[0].minor_axis_length

    # Major Axis Length: 
    shapeFeatVector[7] = Feat[0].major_axis_length

    #Ratio
    shapeFeatVector[8] = Feat[0].minor_axis_length/Feat[0].major_axis_length

    # Solidity: 
    shapeFeatVector[9] =Feat[0].solidity

    return shapeFeatVector


def fun_sigmoid(theta, X):
    p=np.dot(theta,np.transpose(X))
    g=1/(1+np.exp(-1*p))
    return g


def fCalculateCostLogReg(y, y_hat) :

    cost_i=y*np.log(y_hat)+(1-y)*np.log(1-y_hat)
    return cost_i

def fTrain_LogisticReg(X_train, Y_train, alpha) :
    #initialisation
    VERBOSE = True;
    max_iter = 100; 
    
    m,n = X_train.shape;
    h_train=np.zeros((1,m))   
   
    #the cost
    J = np.zeros(max_iter+1);
    #the thetas
    theta = np.zeros((1,n+1));

# *************************************************************************
    x_i=np.zeros((m,n+1))
    x_i[:,0]=1
    total_cost = 0;
    for i in range(m):
        x_i[i,1:n+1]  =X_train[i, :] 
        h_train[0,i]=fun_sigmoid(theta,x_i[i])

        total_cost =total_cost+fCalculateCostLogReg(Y_train[i], h_train[0,i])

    
    # b. Calculate the total cost
    total_cost=(-1/m)*total_cost
    J[0]=total_cost


# *************************************************************************
# GRADIENT DESCENT ALGORITHM TO UPDATE THE THETAS
    
    for num_iter in range(max_iter):
        # STEP 1. Calculate the value of the function h with the current theta values FOR EACH SAMPLE OF THE TRAINING SET 
        gradient=np.zeros((m,n+1))
        x_i=np.zeros((m,n+1))
        x_i[:,0]=1
        for i in range(m):
            x_i[i,1:n+1] =X_train[i, :] 
            h_train[0,i]=fun_sigmoid(theta, x_i[i])
            gradient[i,:]=(h_train[0,i]-Y_train[i])*x_i[i]
        
        # STEP 2. Update the theta values.
        theta=theta-alpha*(1/m)*sum(gradient)

        # STEP 3. Calculate the cost on this iteration and store it on vector J.
        x_i=np.zeros((m,n+1))
        x_i[:,0]=1
        total_cost=0
        for i in range(m):
            x_i[i,1:n+1]  =X_train[i, :]
            h_train[0,i]=fun_sigmoid(theta,x_i[i])
            total_cost += fCalculateCostLogReg(Y_train[i], h_train[0,i])
       
        total_cost=(-1/m)*total_cost
        J[num_iter+1]=total_cost

    if VERBOSE:
        
        plt.plot(range(len(J)), J, '-')
        plt.title('Cost function ')
        plt.xlabel('Number of iterations');
        plt.ylabel('Cost J');
        
    return theta,J


def fClassify_LogisticReg(X_test, theta):
    m,n= X_test.shape;
    y_hat = np.zeros((m));
    x_test_i=np.zeros((m,n+1))
    x_test_i[:,0]=1
    for i in range(m):
        x_test_i[i,1:n+1]  =X_test[i, :]
        y_hat[i] = fun_sigmoid(theta, x_test_i[i])
    return y_hat


def ROC(Y_test,Y_test_hat):
    
    L=np.arange(0,1,0.001)
    T=[]
    F=[]
    for j in L:
        Ypred=Y_test_hat>=j
   #matrice de confusion
        M=np.zeros((2,2))
        for i in range (len(Y_test)):
            if Y_test[i]==True and Ypred[i]==True:
                M[0,0]+=1
            elif Y_test[i]==False and Ypred[i]==True:
                M[1,0]+=1                
            elif Y_test[i]==True and Ypred[i]==False:
                M[0,1]+=1                
            elif Y_test[i]==False and Ypred[i]==False:
                M[1,1]+=1

        tpr=M[0,0]/(M[0,0]+M[0,1])
        fpr=M[1,0]/(M[1,0]+M[1,1])
        T.append(tpr)
        F.append(fpr)
    return T,F