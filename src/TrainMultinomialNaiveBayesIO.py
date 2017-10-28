import sys;
import numpy as np
from collections import Counter
from _ctypes_test import func
import math


class TrainMultinomialNaiveBayes:
    
# |---------------------------------------------------------------|
# trainNaiveBayes
# |---------------------------------------------------------------|
    def trainNaiveBayes(self,classlist,NoOfDocsInClass,totalDocs,totalTermsInAllClasses):
        """
        This method calculates prior probability of each class and conditional probability for each term for each class
        """
        priorProb=np.array([])
        NoOfClasses=len(classlist)
        condProbList=[]
        
        
        for classIndex,classk in enumerate(classlist):
            Nc=NoOfDocsInClass[classIndex]
            priorProb=np.append(priorProb, (float(Nc)/float(totalDocs)))
#                 #debug
#                 print("priorProb : {}".format(priorProb))
#                 #debug -ends
            myclassdict = Counter(classk)
            classset = set(classk)
            totalTermsinClass=len(classk)
            
            
            TotalOfThisTerm=np.array([])
            
           
            
            #debug
            print("classset : {}".format(classset))
            #debug -ends
            for term in classset:
                TotalOfThisTerm=np.append(TotalOfThisTerm,myclassdict[term])
                
            #debug
            print("TotalOfThisTerm : {}".format(TotalOfThisTerm))
            #debug -ends
       
            TermAndProbList=[]
            for termIndex,term in enumerate(classset):
                #debug
                print('----------------------------------------')
                print("termIndex : {}, term: {}".format(termIndex, term))
                #debug -ends
                condProbability= (TotalOfThisTerm[termIndex]+1)/(totalTermsinClass+totalTermsInAllClasses)
                TermAndProbList.append(term)
                TermAndProbList.append(condProbability)
                #debug
                print("TermAndProbList : {}".format(TermAndProbList))
                print('----------------------------------------')
                #debug -ends
            #for termIndex, term -ends
            condProbList.append(TermAndProbList)
        #for classIndex,classk ends       
                
            #debug
            print("condProbList : {}".format(condProbList))
            #debug -ends
                
        return priorProb,condProbList,NoOfClasses;
#|--------------------------trainNaiveBayes -ends-------------------|

# |---------------------------------------------------------------|
# applyMultinomialNaiveBayes
# |---------------------------------------------------------------|
    def applyMultinomialNaiveBayes(self, NoOfClasses,priorProb, condProbList, TestVocab):
        """
        This method takes TestVocab for a document and predicts the class in which it may belong to
        """
        score=np.array([])
        
        for classIndex in range(0,NoOfClasses):
#                 #debug
#                 print("priorProb[classIndex] : {}".format(priorProb[classIndex]))
#                 #debug -ends
            newscore=float(math.log(priorProb[classIndex]))
            print("-------------------------------------------------------------------------------")
            for term in TestVocab:
                #debug
                print("condProbList[classIndex][termIndex] : {}".format(condProbList[classIndex][1]))
                #debug -ends
                #debug
                print("classIndex : {}".format(classIndex))
                #debug -ends
                #debug
                print("term : {}".format(term))
                #debug -ends
                newscore=float(newscore)+float(math.log(condProbList[classIndex][1]))
            #for ends
            score=np.append(score,newscore)
        
        #for ends 
        #debug
        print("score : {}".format(score))
        #debug -ends
        predictedClass=np.argmax(score)
        #debug
        print("predictedClass : {}".format(predictedClass))
        #debug -ends
                   
                
                    
            
            
            
                
                
                
        
    #|--------------------------applyMultinomialNaiveBayes -ends-------------------|
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    trainingFolderPath="../dataset/Training"
    trainMultinomialNaiveBayes = TrainMultinomialNaiveBayes()
    classlist=[]
   
    classlist.append(['Chinese','Beijing','Chinese','Chinese', 'Chinese','Shanghai','Chinese','Macao'])
    classlist.append(['Tokyo', 'Japan', 'Chinese'])
    
    NoOfDocsInClass=[3,1]
    totalDocs=4
    TestVocab=['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan']
    
    priorProb,condProbList,NoOfClasses=trainMultinomialNaiveBayes.trainNaiveBayes(classlist,NoOfDocsInClass,totalDocs)
    
    #debug
    print("condProbList : {}".format(condProbList))
    #debug -ends
    
    predictedClass=trainMultinomialNaiveBayes.applyMultinomialNaiveBayes(NoOfClasses,priorProb,condProbList,TestVocab)
    
    
    