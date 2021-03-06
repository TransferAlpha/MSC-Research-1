# OCIL instance by python

import numpy as np
import random
import math
import copy
from collections import Counter

def check(data): #input:data matirx(two dims) output:the index list of the categorical and numerical attributes
    m, n = data.shape
    N = []
    C = []

    for i in range(n):
        try:
            if isinstance(data[0,i],int) or isinstance(data[0,i],float):
                N.append(i)
            elif isinstance(data[0,i],str):
                C.append(i)
            else:
                raise ValueError("the %d column is not a number or a string column" % i)
        except TypeError as te:
            print(te)
    
    return C, N

def entropyOCIL(dataC): # get the entropy of each Ar, input: whole data or categorical data, output: the list
    entropyList = []
    freqList = []
    parameterList = []
    entropyAttriList = []
    frequencyAttri = {}
    categRawData = dataC
    
    for col in range(dataC.shape[1]):
        freqList = []
        entropyList = []
        frequencyAttri = dict(Counter(categRawData[:, col]))
        for i in frequencyAttri:
            entropyList.append(frequencyAttri[i] / len(categRawData[:, col]))
        for i in range(len(entropyList)):
            freqList.append(-entropyList[i]*math.log(entropyList[i]))
        entropyAttriList.append(sum(freqList)/len(frequencyAttri.items()))
    for i in range(len(entropyAttriList)):
        parameterList.append(entropyAttriList[i]/sum(entropyAttriList))
    
    return parameterList

def similarityCategorical(Categorical, cluster, entropy):#get the similarity between CateAttri input:Categorical array, cluster array，entropy list. output:the similarity between two arrays, float
    hemingwayDis = []
    simiCategorical = np.array([])
    for i in range(len(Categorical)):
        if Categorical[i] == cluster[i]:
            hemingwayDis.append(1)
        else:
            hemingwayDis.append(0)
    simiCategorical = np.multiply(np.array(hemingwayDis), np.array(entropy))

    return sum(simiCategorical.tolist())

def similarityNumerical(Numerical, cluster,clusterTarget):#get the similarity of NumAttri in input:Numerical array,cluster:matrix of the cluster center,clustertarget: the target cluster center output:similarity float
    
    distanceList = []
    for i in range(len(cluster)):
        distanceList.append(math.exp(-0.5*sum((Numerical - cluster[i,:])**2)))

    return math.exp(-0.5*sum((Numerical-clusterTarget)**2))/sum(distanceList)


                  
def clusterFinder(dataC, dataN, clusterC, clusterN, dc, dn,):#get the clusterlabel of each data samples, input:data,cluster,dc:categorical weight dn:numerical weight,output: the list of clusterlabel
    
    similarity = []
    clusterLabel = []
    entropyInstance = entropyOCIL(dataC)
    for i in range(dataC.shape[0]):
        similarity = []
        
        for j in range(clusterC.shape[0]):
           similarity.append(dc*similarityCategorical(dataC[i,:], clusterC[j,:], entropyInstance)+dn*similarityNumerical(dataN[i,:], clusterN, clusterN[j,:]))
        clusterLabel.append(similarity.index(max(similarity)))
    return clusterLabel   

def clusterUpdate(dataC, dataN, clusterLabel):#get the new cluster, input:data matrix, clusterlabel output: the new cluster matrix
    clusterContainer = []
    attributeContainer = []
    clusterContainerN = []
    attributeContainerN = []
    clusterNew = np.zeros((len(Counter(clusterLabel)),dataC.shape[1]),dtype = object)
    clusterNewN = np.zeros((len(Counter(clusterLabel)),dataN.shape[1]),dtype = float)
    

    for i in range(len(Counter(clusterLabel))):
        clusterContainer = []
        for k in range(dataC.shape[1]):
            attributeContainer = []
            for j in range(dataC.shape[0]):
                if clusterLabel[j] == i:
                    attributeContainer.append(dataC[j,k])
            clusterContainer.append(Counter(attributeContainer).most_common(1)[0][0])
        clusterNew[i,:] = copy.deepcopy(clusterContainer)
    
    
    for i in range(len(Counter(clusterLabel))):
        clusterContainerN = []
        for k in range(dataN.shape[1]):
            attributeContainerN = []
            for j in range(dataN.shape[0]):
                if clusterLabel[j] == i:
                    attributeContainerN.append(dataN[j,k])
            clusterContainerN.append(sum(attributeContainerN)/len(attributeContainerN))
        clusterNewN[i,:] = copy.deepcopy(clusterContainerN)
    
    
    return clusterNew, clusterNewN
                    
def iterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, dc):#do the iterative work input:data and cluster,iterative max times output:the clusterlabel and the clusterArray
    dn = 1-dc
    clusterLabel = []
    clusterLabelOld = []
    clusterCNew = copy.deepcopy(clusterC)
    clusterNNew = copy.deepcopy(clusterN)
    for i in range(iteramax):
        clusterLabel = clusterFinder(dataC, dataN, clusterCNew, clusterNNew, dc, dn)
        if clusterLabel == clusterLabelOld:
            #print("final iterative times: ",i)
            break
        clusterLabelOld = copy.deepcopy(clusterLabel)
        clusterCNew, clusterNNew = clusterUpdate(dataC, dataN, clusterLabel)
    
    #print("final iterative timesL ", iteramax)

    return clusterCNew, clusterNNew, clusterLabel





if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from collections import Counter
    import matplotlib.pyplot as plt

    iris = load_iris()
    dc = 0.0
    itera = 0
    clusterlabel = []
    k = int(input("input the cluster number: "))
    klist = [30,80,120]
    clusterNNew = iris.data[klist,:]
    clusterCate = iris.target.reshape(150,1)
    clusterCNew = copy.deepcopy(clusterCate[klist,:])
    itera = int(input("max iterative times "))
    clusterCcontainer = np.zeros((k,iris.data.shape[1]),dtype = object)
    clusterNcontainer = np.empty((k,iris.data.shape[1]),dtype = float)
    weightList = np.arange(0,0.4,0.01)
    accuracyRate = []
    for i in range(len(weightList)):
        clusterCcontainer, clusterNcontainer, clusterlabel = iterativeFunc(clusterCate, iris.data, clusterCNew, clusterNNew, itera, weightList[i])
        accuracyRate.append(sum(clusterlabel == iris.target)/len(iris.target))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set(xlim=[weightList[0],weightList[-1]],ylim=(accuracyRate[0],accuracyRate[-1]+0.02), title='the image of weight function',
            ylabel='the accuracy', xlabel='weight of categorical attritubes')
    ax1.plot(weightList,accuracyRate)
    plot.show()

    clusterCcontainer, clusterNcontainer, clusterlabel = iterativeFunc(clusterCate, iris.data, clusterCNew, clusterNNew, itera, 0.01)
    cluster1 = {}
    for i in range(len(Counter(clusterlabel))):
        clustering = []
        for j in range(len(clusterlabel)):
            if clusterlabel[j] == i:
                clustering.append(j)
        cluster1[i] = clustering
    data1 = iris.data[cluster1[0],:]
    data2 = iris.data[cluster1[1],:]
    data3 = iris.data[cluster1[2],:]
    plt.scatter(data1[:,0],data1[:,1],marker = 'x',color = 'red', s = 50,label = 'first')
    plt.scatter(data2[:,0],data2[:,1],marker = '+',color = 'blue', s = 50,label = 'second')
    plt.scatter(data3[:,0],data3[:,1],marker = 'o',color = 'green', s = 50,label = 'third')
    plt.title('the distribution when dc = 0')
    plt.show()


    




    
    




    













            



