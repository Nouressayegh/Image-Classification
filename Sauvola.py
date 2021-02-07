import numpy as np
import matplotlib . pyplot as plt 
from skimage import data

#chargement de l'image et l'affichage de son histogramme
image = data.page()
plt.hist(image.flatten(),256)
plt.show()
plt .imshow(image, plt . cm.gray);
plt.axis('off')
plt.show() 


#on défini la fonction qui prend l'image et la fenêtre de pixel et retourne une image binarisée
def seuil_sauvola(I,a,k):
    
    #les constantes
    L=a//2
    a2=a*a
    R=128
    image = np.copy(I)
    
    #Pour éviter les problèmes de bords on ajoute des bordures de largeur L
    image=np.pad(image,L, mode='edge')
    
    #les dimensions de l'image
    mx,my=image.shape
    
    #initialisation des élèments pour le calcul des sommes
    som1=[0 for i in range (my)]
    som2=0
    mean1=np.zeros((mx,my));
    
    #moyenne
    m=np.zeros((mx,my));
    #variance
    s2=np.zeros((mx,my));
    #seuil
    T=np.zeros((mx,my));
    
    
    #remlissage de T 
    for x in range(L,mx-L):
        for y in range(L,my-L):
            
            #calcul de la moyenne
            mean1[x,y]=sum( [image[i,y]/a for i in range (x-L,x+L+1)])
            m[x,y]=sum( [mean1[x,j]/a for j in range(y-L,y+L+1)]) 
                    
            #calcul de la variance
            som1[y]=sum([(image[i,y]-m[x,y])**2 for i in range(x-L,x+L+1)])
            som2=sum([som1[j] for j in range(y-L,y+L+1)])
            s2[x,y]=som2/a2 
            
            #ecart-type
            s=np.sqrt(s2)
            
            #calcul du seuil
            T[x,y]=m[x,y]*(1+k*(s[x,y]/R-1))
            
            #binarisation
            if image[x,y]>T[x,y]:
                image[x,y]=255
            else:
                image[x,y]=0
                
    image=image[L:mx-L,L:my-L]
    return image

#affichage de l'image
plt .imshow(seuil_sauvola(image,15,0.2), plt . cm.gray);
plt.axis('off')
plt.show()





