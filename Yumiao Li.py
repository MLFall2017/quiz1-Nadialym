# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:42:42 2017

@author: lymna
"""

#1 calculate the variance of every variable in the data file.
import csv
import numpy as np
import matplotlib.pyplot as plt


#1. Taking the whole dataset 
x=[]
y=[]
z=[]

with open('dataset_1.csv', 'rb') as f:
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        x.append( float(row[0]))
        y.append( float(row[1]))
        z.append( float(row[2]))
        
var_x=np.var(x)
var_y=np.var(y)
var_z=np.var(z)

#2 calculate the covariance
cov_xy=np.cov(x,y)
cov_yx=np.cov(y,z)

#PCA
 #2 mean centering 
mean_x=np.mean(x)
mean_y=np.mean(y)
mean_z=np.mean(z)

xc=x-mean_x
yc=y-mean_y
zc=z-mean_z


X=np.array(xc)
Y=np.array(yc)
Z=np.array(zc)


my_var=np.var(x)

#3 calculate Cov_matrix
mean_vec = np.stack((xc, yc, zc),axis=1)
cov_matrix = np.cov(mean_vec,rowvar=False)
print mean_vec

#4 calculate eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)#eigh for sysmmatric matrice

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#5 select principle components
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

matrix_p=np.stack((eig_vecs[:,0],eig_vecs[:,1],eig_vecs[:,2]),axis=1)

print('Matrix P:\n', matrix_p)


N=mean_vec.dot(matrix_p)

# 6Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

#5 Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


indices=np.argpartition(eig_vals,-2)[-2:]
plt.scatter(np.array(N[:,indices[1]]),np.array(N[:,indices[0]]))
plt.title("PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

#----------------------------------------------------------------------------
#2.(2)
a = np.array([[0, -1], [2, 3]], float)
eig_valsA, eig_vecsA = np.linalg.eig(a)

