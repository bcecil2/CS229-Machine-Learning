import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import genfromtxt

def h(xVec,thetas):
    return np.dot(np.transpose(thetas),xVec)

def J(xs,ys,thetas,size):
    scale = .5
    return scale*sum([(h(xs[i],thetas)-ys[i])**2 for i in range(size)])

def LMS(xs,ys,thetas):
    alpha = .01
    m,n = xs.shape
    
    for k in range(10000):
        for j in range(n):
            s = 0
            for i in range(m):
                s += (ys[i] - h(xs[i],thetas))*xs[i][j]
            s *= alpha
            thetas[j] += s

    return thetas


def SGD(xs,ys,thetas):
    alpha = .01
    m,n = xs.shape
    for k in range(10000):
        for i in range(m):
            for j in range(n):
                grad = alpha*(ys[i] - h(xs[i],thetas))*xs[i][j]
                thetas[j] += grad
    return thetas

def normalEq(xs,ys):
    return np.linalg.inv(np.transpose(xs)@xs)@np.transpose(xs)@ys

def normalize(x):
    xMax = x.max()
    xMin = x.min()
    return (x - xMax)/(xMax - xMin)

data = genfromtxt("Housing.csv", delimiter=",")
price = [ [1,data[i][1]] for i in range(len(data))] 
price = price[1:]


size = [ data[i][2] for i in range(len(data))]
size = size[1:]


xTrain = np.array(price[:-250])
xTest = np.array(price[-250:])

yTrain = np.array(size[:-250])
yTest = np.array(size[-250:])

xTrain = normalize(xTrain)
xTest = normalize(xTest)
# dimension 
d = len(xTrain[0]) 
# num samples
m = len(xTrain)



thetas = np.zeros(d)

#params = LMS(xTrain,yTrain,thetas)

#print(params)

#params = SGD(xTrain,yTrain,thetas)
params = normalEq(xTrain,yTrain)
print(params)
x = [j[1] for j in xTest]

est = [h(xTest[i],params) for i in range(len(xTest))]


plt.scatter(x,yTest,color='black')
plt.title("Housing Prices Based on Size")
plt.xlabel("Size")
plt.ylabel('Price')
plt.plot(x, est)
plt.show()

