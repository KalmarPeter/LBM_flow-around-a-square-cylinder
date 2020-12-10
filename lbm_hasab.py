# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:03:59 2020

@author: peter
"""
from numpy import*
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import njit
import pandas as pd

start = time.time()

#Konstansok:

##Szimulaciós beáll.

mIter = 350000
uLB = 0.04 #Maximális szimulációs áramlási sebesség (Stabilitás miatt)



##Domain def.

nx = 840 #Cellaszám x irányban
ny = 360 #Cellaszám y irányban
h = ny//10 #Hasáb oldalhosszai (egyben jellemző méret)
hx = nx // 5 #Hasáb x irányú pozíciója
hy = (ny //2 - h//2)+3 #Hasáb y irányú pozíciója a szimetria elkerülendő
l = ny-1 #Domain magassága (cellában)
px = (nx//5) + h + 4*h
py = (ny //2 - h//2) + 3

##Dimenziótlan számok:

Re = 125 #Reynolds szám
nuLB = (uLB *h)/Re #Kinematikai viszkozitás (modell)
lamb = 0.25  #lambda TRT-hez

##Lattice konst.

v = array([ [ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],[ 0, -1], [-1,  1], [-1,  0], [-1, -1] ]) #Lattice sebességek
w = array([ 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36]) #Lattice sebességek súlyozása
omega = 1 / (3*nuLB +0.5) #Relaxációs paraméter (fordítottan arányos a viszkozitással)
fin = zeros((9,nx,ny))
    
#Hasáb geometria + falak allul és fellül:

def nemfolyadek(i, j):
    szilard=zeros((i,j),dtype=bool)
    szilard[:,0] = True
    szilard[:,j-1] = True
    for x in range(i):
        for y in range(j):
            if ((hy <= y) & (y< hy+h)) & ((hx <= x) & (x < hx+h)):
                szilard[x,y]=True
    return szilard

szilard=nemfolyadek(nx,ny)

#Függvényekkel gyorsabb az élet...

##Makroszkópikus változók:
@njit
def makrovalt(fin):
    rho = sum(fin, axis=0) 
    u = zeros((2, nx, ny)) #u=(1/rho)*(sum(i=0->8) vi*fi) Makroszkopikus sebesség cellánként 
    for i in range(9):
        u[0,:,:] += v[i,0] * fin[i,:,:]
        u[1,:,:] += v[i,1] * fin[i,:,:]
    u /= rho
    return rho, u

###Equilibrum distribution function (Egyensúlyi eloszlás fv):
@njit
def edf(rho,u):
    feq = zeros((9,nx,ny))
    unegyzet = 3/2 * (u[0]**2 + u[1]**2)
    for i in range(9):
        vu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
        feq[i,:,:] = rho*w[i] * (1 + vu + 0.5*vu**2 - unegyzet)
    return feq

###Makroszkópikus (initial) sebességprofil kialakítása az összes cellában:

def initialu(uLB,nx,ny):   
    u=zeros((2,nx,ny))
    for y in range(ny):
        u[0,:,y]=(4*uLB / (l*l)) * y*(l-y)
    return u
  
#Kezdeti értékek betöltése:

##Meghívva kezdu-néven:

kezdu = initialu(uLB,nx,ny)

##Kezdeti Eloszlás fv definiálása =  (Egyensúlyi eloszlás fv (sűrűség, kezdeti sebesség)):

fin = edf(1,kezdu) #Minnél több ciklus van annál inkább alhanyagolhatóbb a jelentősége

#Collision TRT model:

@njit
def colltrt(fin,feq):   
    omegan = 1 / (( lamb / (( 1 / omega) -0.5 )) + 0.5)
    fout = zeros((9,nx,ny))
    for i in range(9):
        fout[i,:,:] = fin[i,:,:] - omega*((fin[i,:,:] + fin[8-i,:,:])/2 - (feq[i,:,:] + feq[8-i,:,:])/2) - omegan*((fin[i,:,:] - fin[8-i,:,:])/2 - (feq[i,:,:] - feq[8-i,:,:])/2)
    
    return fout

#Streaming:

@njit
def streamlbm(fout,v,fin):
    for x in range(nx):
        for y in range(ny):
            for i in range(9):
                kovx = x + v[i,0]
                kovy = y + v[i,1]
                
                if kovx < 0:
                    kovx = nx-1                   
                if kovx >= nx:
                    kovx = 0
                    
                if kovy < 0:
                    kovy = ny-1                   
                if kovy >= ny:
                    kovy = 0
                    
                fin[i,kovx,kovy] = fout[i,x,y]
    
    return fin

#Mérési pontokon lévő sebesség és nyomás allokálása:

sebesseg = []
nyomas = []




# A főciklus

for t in range(mIter):
    
    
    # Makroszkopikus változók:
      
     rho, u = makrovalt(fin)
     
    # Makroszkópikus változók átírása a peremeken:
        
    ## Inlet Zou and He makroszkopikus Piseuille profil: 
    
     for y in range(ny-2):
         u[0,0,y+1]=(4*uLB / (l*l)) * (y+1)*(l-(y+1))
         u[1,0,y]=0
         
     for y in range(ny-2):
         rho[0,y+1]=(1/(1-u[0,0,y+1]))*((fin[3,0,y+1]+fin[4,0,y+1]+fin[5,0,y+1])+2*(fin[6,0,y+1]+fin[7,0,y+1]+fin[8,0,y+1]))
     
    ## Outlet Zou and He makroszkópikus Constans nyomás:
        

     for y in range(ny-2):
         rho[nx-1,y+1] = 1
         u[1,nx-1,y+1] = 0
         
     for y in range(ny-2):    
         u[0,nx-1,y+1] = -1 + (1/(rho[nx-1,y+1]))*((fin[4,nx-1,y+1] + fin[3,nx-1,y+1] + fin[5,nx-1,y+1]) + 2*(fin[1,nx-1,y+1] + fin[0,nx-1,y+1] + fin[2,nx-1,y+1]))
        
    #Mikroszkópikus Peremfeltételek a peremeken:
        
    ## Inlet Zou and He mikroszkopikus Poiseuille peremfeltétel:
        
     for y in range(ny-2):
         fin[1,0,y+1] = fin[7,0,y+1] - (2/3)*rho[0,y+1]*u[0,0,y+1]
         fin[0,0,y+1] = fin[8,0,y+1] + (1/2)*(fin[5,0,y+1] - fin[3,0,y+1]) + (1/2)*rho[0,y+1]*u[1,0,y+1] + (1/6)*rho[0,y+1]*u[0,0,y+1]
         fin[2,0,y+1] = fin[6,0,y+1] + (1/2)*(fin[3,0,y+1] - fin[5,0,y+1]) + (1/2)*rho[0,y+1]*u[1,0,y+1] + (1/6)*rho[0,y+1]*u[0,0,y+1]
         
    ## Outlet Zou and He mikroszkopikus Constans nyomás:
        
     for y in range(ny-2):
         fin[7,nx-1,y+1] = fin[1,nx-1,y+1] - (2/3)*rho[nx-1,y+1]*u[0,nx-1,y+1]
         fin[8,nx-1,y+1] = fin[0,nx-1,y+1] + (1/2)*(fin[3,nx-1,y+1] - fin[5,nx-1,y+1]) + (1/2)*rho[nx-1,y+1]*u[1,nx-1,y+1] + (1/6)*rho[nx-1,y+1]*u[0,nx-1,y+1]
         fin[6,nx-1,y+1] = fin[2,nx-1,y+1] + (1/2)*(fin[5,nx-1,y+1] - fin[3,nx-1,y+1]) + (1/2)*rho[nx-1,y+1]*u[1,nx-1,y+1] + (1/6)*rho[nx-1,y+1]*u[0,nx-1,y+1]   
     
    #Egyensúlyi eloszlás fv. számítása:
    
     feq=edf(rho,u)
          
    #Collision TRT model:
    
     fout = colltrt(fin,feq)
         
    #Bounce-back cellákban más ütközési szbály kialakítása:
        
     for i in range(9):
         fout[i,szilard] = fin[8-i,szilard]
         
    #Streaming:
     
     fin = streamlbm(fout,v,fin)
     
    #Sebesség és nyomás a mérési pontban 
    
     seb = (u[0,px,py]**2 + u[1,px,py]**2)**(1/2)
     nyom = (1/3)*rho[px,py]
     sebesseg.append(seb)
     nyomas.append(nyom)
    
    #Sebesség mező vizualizációja:
   
     if t%100000 ==0:
         
        plt.close()
        plt.imshow(sqrt(u[0]**2+u[1]**2).transpose(), cmap=cm.Blues)
        plt.colorbar()
        plt.xlabel("nx [lattice cella]")
        plt.ylabel("ny [lattice cella]")
        plt.savefig("re{:01}.png".format(t))


#Sebesség és nyomás értékek importálása excel-be:
    
dict = {'Sebesseg' : sebesseg, 'Nyomas' : nyomas}
df = pd.DataFrame(dict)
df.to_excel('Re125kozepeshalo11.0.xlsx')
end = time.time()
print(end-start)