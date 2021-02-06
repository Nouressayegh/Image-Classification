import numpy as np
import skimage.measure as sm
import skimage.transform



def feret_diameter(I):
    d = np.max(I.shape)
    D = 0

    for a in np.arange(0, 180, 30):
        I2 = skimage.transform.rotate(I, angle=a, order=0)
        F = np.max(I2, axis=0)
        measure = np.sum(F )

        if (measure < d):
            d = measure
        if (measure > D):
            D = measure
    return d, D

def roundness(I,A,D):
    return(4*A)/(np.pi*D**2)

def circularity(I,A,P):
    return (4*np.pi*A)/P**2
    
def fGetShapeFeat(region):
    d,D=feret_diameter ( region )
    shapeFeatVector =np.zeros(9);
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
    #difference between the ferets diameters
    shapeFeatVector[5]=D-d
    #elongation
    elongation=d/D
    shapeFeatVector[6]=elongation
    #Roundness
    shapeFeatVector[7]=roundness(region,Feat[0].area,D)
    #Circularity
    shapeFeatVector[8]=roundness(region,Feat[0].area,Feat[0].perimeter)
    return shapeFeatVector

def lbp(I):
    nx,ny=I.shape
    win=np.array([[1,2,4], [8,0,16],[32,64,128] ])
    lbp=np.zeros((nx,ny))
    for i in np.arange(1,nx-2):
        for j in np.arange(1,ny-2):
            
            #the window
            w=I[i-1:i+2,j-1:j+2]
            w=w>=I[i,j]
            w=w*win
            lbp[i,j]=np.sum(w)
    h,edges=np.histogram(lbp[1:-1,1:-1],density=True,bins=255)
    return h

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

def AUC(Y_test,Y_test_hat):
    L=np.arange(0,1,0.001)
    T,F=ROC(Y_test,Y_test_hat)
    m=len(L)
    AUC=0
    for i in range(m-1):
        AUC+=(F[i]-F[i+1])*T[i]
    return AUC