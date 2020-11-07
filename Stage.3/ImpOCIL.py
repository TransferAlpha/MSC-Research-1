# OCIL instance by python

import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
from sklearn import preprocessing
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
        distanceList.append(np.exp(-0.5*np.sum(np.square((Numerical - cluster[i,:])))))
    return np.exp(-0.5*np.sum(np.square(Numerical-clusterTarget)))/np.sum(distanceList)


                  
def clusterFinder(dataC, dataN, clusterC, clusterN, dc, dn,):#get the clusterlabel of each data samples, input:data,cluster,dc:categorical weight dn:numerical weight,output: the list of clusterlabel
    
    similarity = []
    clusterLabel = []
    entropyInstance = entropyOCIL(dataC)
    for i in range(dataC.shape[0]):
        similarity = []
        
        for j in range(clusterC.shape[0]):
           similarity.append(dc*similarityCategorical(dataC[i,:], clusterC[j,:], entropyInstance)+dn*similarityNumerical(dataN[i,:], clusterN, clusterN[j,:]))
        clusterLabel.append(similarity.index(np.max(similarity)))
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
#targetLabel is an one dis array or list
def weightNumerical(h,dataN,clusterLabel,numberOfCluster): #h:float, get the weight list of the numerical attributes, return array
    weightOfNumerical = np.ones((numberOfCluster,dataN.shape[1]),dtype=float)/np.sqrt(dataN.shape[1])

    if clusterLabel == []:
        return weightOfNumerical
    weightOfNumerical = np.ones((len(Counter(clusterLabel)),dataN.shape[1]),dtype=float)/np.sqrt(dataN.shape[1])
    indexCluster = []
    for i in range(len(Counter(clusterLabel))):
        for p in range(len(clusterLabel)):
            if clusterLabel[p] == i:
                indexCluster.append(p)
        clusterCon = dataN[indexCluster,:]
        varienceCluster = np.var(clusterCon,axis = 0)
        parameterOne = np.sqrt(np.sum(np.exp(-h*2*varienceCluster)))
        for j in range(weightOfNumerical.shape[1]):
            weightOfNumerical[i,j] = np.exp(-h*varienceCluster[j])/parameterOne
    return weightOfNumerical

def improvedSimilarityNumerical(Numerical, cluster,clusterTarget,weightOfNumerical):#get the similarity of NumAttri in input:Numerical array,cluster:matrix of the cluster center,clustertarget: the target cluster center output:similarity float
#Update about the Stage.3: I updated the clusterTarget into the index of target center, not the array of center, this is more reasonable
    distanceList = []
    #for i in range(len(cluster)):
        #distanceList.append(np.exp(-0.5*np.sqrt(np.sum(np.square((Numerical - cluster[i,:]))*weightOfNumerical[i,:]))))
    for i in range(len(cluster)):

        distanceList.append(np.exp(-0.5*np.sum(np.square(Numerical - cluster[i,:])*weightOfNumerical[i,:])))
    
    return np.exp(-0.5*np.sum(np.square(Numerical-cluster[clusterTarget,:])*weightOfNumerical[clusterTarget,:]))/np.sum(distanceList)


def improvedClusterFinder(dataC, dataN, clusterC, clusterN, dc, dn, weightOfNumerical):
    similarity = []
    clusterLabel = []
    entropyInstance = entropyOCIL(dataC)
    for i in range(dataC.shape[0]):
        similarity = []
        
        for j in range(clusterC.shape[0]):
            #print("Start",i,dataN,dataN[i,:], clusterN,"end")
            similarity.append(dc*similarityCategorical(dataC[i,:], clusterC[j,:], entropyInstance)+dn*improvedSimilarityNumerical(dataN[i,:], clusterN, j, weightOfNumerical))
        clusterLabel.append(similarity.index(np.max(similarity)))
    return clusterLabel   

def improvedIterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, dc, h):#do the iterative work input:data and cluster,iterative max times output:the clusterlabel and the clusterArray
    dn = 1-dc
    clusterLabel = []
    clusterLabelOld = []
    clusterCNew = copy.deepcopy(clusterC)
    clusterNNew = copy.deepcopy(clusterN)
    weightOfNumerical = weightNumerical(h,dataN,clusterLabel,clusterN.shape[0])
    for i in range(iteramax):
        clusterLabel = improvedClusterFinder(dataC, dataN, clusterCNew, clusterNNew, dc, dn, weightOfNumerical)
        weightOfNumerical = weightNumerical(h,dataN,clusterLabel,clusterN.shape[0])
        if clusterLabel == clusterLabelOld:
            #print("final iterative times: ",i)
            break
        clusterLabelOld = copy.deepcopy(clusterLabel)
        clusterCNew, clusterNNew = clusterUpdate(dataC, dataN, clusterLabel)
    
    #print("final iterative timesL ", iteramax)

    return clusterCNew, clusterNNew, clusterLabel
def weightOverallNumerical(h,dataN,numberOfCluster):
    weightOfNumerical = np.ones((numberOfCluster,dataN.shape[1]),dtype=float)/np.sqrt(dataN.shape[1])
    varienceClusterOverall = np.var(dataN, axis =0)
    parameterOne = np.sqrt(np.sum(np.exp(-h*2*varienceClusterOverall)))
    for i in range(weightOfNumerical.shape[0]):
        for j in range(weightOfNumerical.shape[1]):
            weightOfNumerical[i,j] = np.exp(-h*varienceClusterOverall[j])/parameterOne
    return weightOfNumerical


def improvedMEIterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, dc, h):#do the iterative work input:data and cluster,iterative max times output:the clusterlabel and the clusterArray
    dn = 1-dc
    clusterLabel = []
    clusterLabelOld = []
    clusterCNew = copy.deepcopy(clusterC)
    clusterNNew = copy.deepcopy(clusterN)
    weightOfNumerical = weightOverallNumerical(h,dataN,clusterN.shape[0])
    for i in range(iteramax):
        clusterLabel = improvedClusterFinder(dataC, dataN, clusterCNew, clusterNNew, dc, dn, weightOfNumerical)
        #weightOfNumerical = weightNumerical(h,dataN,clusterLabel,clusterN.shape[0]) 
        if clusterLabel == clusterLabelOld:
            #print("final iterative times: ",i)
            break
        clusterLabelOld = copy.deepcopy(clusterLabel)
        clusterCNew, clusterNNew = clusterUpdate(dataC, dataN, clusterLabel)
    
    #print("final iterative timesL ", iteramax)

    return clusterCNew, clusterNNew, clusterLabel

#def improvedClusterFinder(dataC):

def accuracyFigure(dataC, dataN, clusterC, clusterN, iteramax, start, stop, step, targetLabel, h):#draw the accuracy figure by modifing the dc, using the step between start and stop
    accuracyRate = []
    improvedAccuracyRate = []
    improvedMEAccuracyRate = []
    clusterlabel = []
    improvedClusterlabel = []
    improvedMEClusterlabel = []
    clusterCNew = copy.deepcopy(clusterC)
    clusterNNew = copy.deepcopy(clusterN)
    improvedClusterCNew = copy.deepcopy(clusterC)
    improvedClusterNNew = copy.deepcopy(clusterN)
    improvedMEClusterCNew = copy.deepcopy(clusterC)
    improvedMEClusterNNew = copy.deepcopy(clusterN)
    weightList = np.arange(start, stop, step).tolist()
    for i in range(len(weightList)):
        clusterCNew, clusterNNew, clusterlabel = iterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, weightList[i])
        #print(clusterlabel == targetLabel)
        sumnumber = 0
        for j in range(len(clusterlabel)):
            if clusterlabel[j] == targetLabel[j]:
                sumnumber = sumnumber+1
        accuracyRate.append(sumnumber/len(targetLabel))
    #print(accuracyRate)
    for i in range(len(weightList)):
        improvedClusterCNew, improvedClusterNNew, improvedClusterlabel = improvedIterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, weightList[i], h)
        #print(clusterlabel == targetLabel
        sumnumber = 0
        for j in range(len(improvedClusterlabel)):
            if improvedClusterlabel[j] == targetLabel[j]:
                sumnumber = sumnumber+1
        improvedAccuracyRate.append(sumnumber/len(targetLabel))
    #print(accuracyRate)
    for i in range(len(weightList)):
        improvedMEClusterCNew, improvedMEClusterNNew, improvedMEClusterlabel = improvedMEIterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, weightList[i], h)
        #print(clusterlabel == targetLabel
        sumnumber = 0
        for j in range(len(improvedMEClusterlabel)):
            if improvedMEClusterlabel[j] == targetLabel[j]:
                sumnumber = sumnumber+1
        improvedMEAccuracyRate.append(sumnumber/len(targetLabel))
    #print(accuracyRate)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set(xlim=[weightList[0],weightList[-1]],ylim=(0,1.02), title='the image of weight function',
            ylabel='the accuracy', xlabel='weight of categorical attritubes')
    ax1.plot(weightList, accuracyRate,label = "the OCIL")
    ax1.plot(weightList,improvedAccuracyRate,label = "Iterative weight")
    ax1.plot(weightList,improvedMEAccuracyRate,label = "Overall weight")
    plt.legend()
    plt.show()

    return 

def scatterShow(dataC, dataN, clusterC, clusterN, iteramax, dc, targetLabel, xcolumn, ycolumn):
    clusterCNew = copy.deepcopy(clusterC)
    clusterNNew = copy.deepcopy(clusterN)
    clusterlabel = []
    colorList = ["darkorange","lightgreen","royalblue","lightcoral","gold","turquoise","crimson","slategray"]
    xData = preprocessing.scale(dataN[:,xcolumn])
    yData = preprocessing.scale(dataN[:,ycolumn])
    clusterCNew, clusterNNew, clusterlabel = iterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, dc)
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set(title = 'The result of clustering')
    ax2 = fig2.add_subplot(111)
    ax2.set(title = 'The result of target')
    for i in range(len(Counter(clusterlabel))):
        dataContainer = []
        for j in range(len(clusterlabel)):
            if clusterlabel[j] == i:
                dataContainer.append(j)
        ax1.scatter(xData[dataContainer], yData[dataContainer], color = colorList[i])
        
    if type(targetLabel) == np.array:
        targetLablelist = targetLabel.tolist()
    else:
        targetLablelist = copy.deepcopy(targetLabel)

    for i in range(len(Counter(targetLablelist))):
        dataContainer = []
        for j in range(len(targetLablelist)):
            if targetLablelist[j] == i:
                dataContainer.append(j)
        ax2.scatter(xData[dataContainer], yData[dataContainer], color = colorList[i])
    plt.show()
    return 
    
def barFigure(dataC, dataN, clusterC, clusterN, iteramax, targetLabel, dc, h, sign):
    accuracyRate = []
    clusterLabel = []
    clusterCN = copy.deepcopy(clusterC)
    clusterNN = copy.deepcopy(clusterN)
    clusterCN, clusterNN, clusterLabel = iterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, dc)
    sumnumber = 0
    for j in range(len(clusterLabel)):
        if clusterLabel[j] == targetLabel[j]:
            sumnumber = sumnumber+1
    accuracyRate.append(sumnumber/len(clusterLabel))
    clusterCN, clusterNN, clusterLabel = improvedIterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, dc, h)
    sumnumber = 0
    for j in range(len(clusterLabel)):
        if clusterLabel[j] == targetLabel[j]:
            sumnumber = sumnumber+1
    accuracyRate.append(sumnumber/len(clusterLabel))
    clusterCN, clusterNN, clusterLabel = improvedMEIterativeFunc(dataC, dataN, clusterC, clusterN, iteramax, dc, h)
    sumnumber = 0
    for j in range(len(clusterLabel)):
        if clusterLabel[j] == targetLabel[j]:
            sumnumber = sumnumber+1
    accuracyRate.append(sumnumber/len(clusterLabel))
    fig, ax1 = plt.subplots(figsize = (6,4))  
    if sign == 2:

        print("The errors of origin algorithm is %f, the improved is %f" %(1-accuracyRate[0], 1-accuracyRate[1]))
    else:
        print("The errors of origin algorithm is %f, the improved is %f, the my method is %f" %(1-accuracyRate[0], 1-accuracyRate[1], 1-accuracyRate[2]))
    xList = np.arange(0,sign,1).tolist()
    colors = ['lightblue','orange','grey']
    ax1.set(title = "The accuracy of the original OCIL and Improved OCIL")
    for i in range(sign):
        ax1.bar(xList[i], accuracyRate[i], align='center', width = 0.2, fc = colors[i])
    if sign == 2:
        ax1.set(xticks = xList, xticklabels = ['Origin','Improved'])
    else:
        ax1.set(xticks = xList, xticklabels = ['Origin', 'Improved', 'ME'])
    plt.show()
    return





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
    weightList = np.arange(0,0.4,0.01).tolist()
    accuracyRate = []
    for i in range(len(weightList)):
        clusterCcontainer, clusterNcontainer, clusterlabel = ImprovedIterativeFunc(clusterCate, iris.data, clusterCNew, clusterNNew, itera, weightList[i])
        accuracyRate.append(sum(clusterlabel == iris.target)/len(iris.target))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set(xlim=[weightList[0],weightList[-1]],ylim=(accuracyRate[0],accuracyRate[-1]+0.02), title='the image of weight function',
            ylabel='the accuracy', xlabel='weight of categorical attritubes')
    ax1.plot(weightList,accuracyRate)
    plt.show()

    clusterCcontainer, clusterNcontainer, clusterlabel = improvedIterativeFunc(clusterCate, iris.data, clusterCNew, clusterNNew, itera, 0.01)
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


    




    
    




    













            



