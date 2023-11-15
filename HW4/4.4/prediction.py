import numpy as np
import math 
from numpy.linalg import inv

with open("stock thing/nasdaq00.txt", "r") as file:
    stock1 = file.readlines()
    file.close()
with open("stock thing/nasdaq01.txt", "r") as file:
    stock2 = file.readlines()
    file.close()
#creating arrays of values 
arr1 = []
arr2 = []
for i in range(len(stock1)):
    temp = (stock1[i].strip())
    arr1.append(float(temp))

for i in range(len(stock2)):
    temp = (stock2[i].strip())
    arr2.append(float(temp))
arr1 = np.array(arr1)
arr2 = np.array(arr2)

def predict(arr):
    predict = arr[3:]
    befores = np.array([arr[2:len(arr)-1],arr[1:len(arr)-2],arr[0:len(arr)-3]])
    A = np.dot(befores, befores.T)
    b = np.dot(befores,predict[:,np.newaxis])
    a_mle = np.dot(inv(A),b)
    return a_mle, predict, befores

def getRMSE(arr):
    totals = []
    best_as, predict, befores = predict(arr)
    return (((predict - np.dot(best_as.T,befores)) ** 2).mean())

print(predict(arr1)[0])
print("2000: " + str(getRMSE(arr1)))
print("2001: " + str(getRMSE(arr2)))
