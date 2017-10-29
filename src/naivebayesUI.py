import sys
import os
import numpy as np
from dataPreprocessing import DataPreprocessing
from TrainMultinomialNaiveBayesIO import TrainMultinomialNaiveBayes

class NaiveBayesUI:
    
#|-----------------------------------------------------------------------------|
# naiveBayesTextClassification
#|-----------------------------------------------------------------------------|
    def naiveBayesTextClassification(self, trainingDirPath, testingDirPath):
        """
        given function performs naiveBayesTextClassification in which it performs
        I. data preprocessing
            1. reading all files
            2. tokenizing content of each files
        II.Naive Bayes 
            1. training data by 
                a. finding prior probability of each class
                b. finding conditional probability of each class based on terms
                c. apply naive Bayes
            2. find prediction class of testing data by 
                a. finding prediction probability score of each class, 
                b. assign clas with max probability score as predicted class
        """
        
        dataPreprocessing = DataPreprocessing()
        print ('STATUS: pre-processing of training data begin:')
        classTokenList,uniqueTokenList, nDocsInClassArr, classNameList = \
                            dataPreprocessing.preprocessTrainingData(\
                                                                trainingDirPath)
        print ('STATUS: pre-processing of training data is done.')
#         #debug
#         print ('classTokenList = {} '.format(classTokenList))
#         print ('uniqueTokenList = {}'.format(uniqueTokenList))
#         print ('nDocInClassArr = {}'.format(nDocsInClassArr))
#         print ('classNameList = {}'.format(classNameList))
#         #debug -ends
         
        totalDocs=np.sum(nDocsInClassArr)
        totalTermsInSllClasses = len(uniqueTokenList)
        trainMultinomialNaiveBayes = TrainMultinomialNaiveBayes()
        print('STATUS: training multinomial naive bayes begin:')
        priorProbArr,condProbList,NoOfClasses=trainMultinomialNaiveBayes.\
                                    trainNaiveBayes(classlist=classTokenList,\
                                                    uniqueTokenList=uniqueTokenList,\
                                               NoOfDocsInClass=nDocsInClassArr,\
                                               totalDocs=totalDocs,\
                                               totalTermsInAllClasses = \
                                                        totalTermsInSllClasses)
        print('STATUS: training multinomial naive bayes done.')
        testingFileTokenDict, fileActualClassDict = \
                                dataPreprocessing.preprocessTestingData(\
                                                testingDirPath = testingDirPath)
#         #debug
#         print ('testingFileTokenDict = {} '.format(testingFileTokenDict))
#         #debug -ends
        print ('STATUS: pre processing of test dataset begin;')
        filePredictedClassDict = trainMultinomialNaiveBayes.classifyAllDocs(\
                                        fileTokenDict = testingFileTokenDict,\
                                        classNameList = classNameList,\
                                        NoOfClasses = NoOfClasses,\
                                        priorProbArr = priorProbArr,\
                                        condProbList = condProbList)
        print ('STATUS: pre processing of test dataset ends.')
#         #debug
#         print ('fileActualClassDict = {} '.format(fileActualClassDict))
#         print ('filePredictedClassDict = {} '.format(filePredictedClassDict))
#         #debug -ends
        print ('STATUS: finding accuracy:')
        predictionAccuracy, correctPrediction, inCorrectPrediction =\
                             trainMultinomialNaiveBayes.findAccuracy(\
                                filePredictedClassDict=filePredictedClassDict,\
                                fileActualClassDict=fileActualClassDict)
        print ('STATUS: Accuracy found')
        return predictionAccuracy, correctPrediction, inCorrectPrediction
#|------------------------naiveBayesTextClassification -ends-------------------|                   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
        
        '''
        This function takes two command line arguments - 
        one is the path of the folder containing training data and 
        the other is the path of the folder containing testing data
        '''
        
        if len(sys.argv)>1:
            trainingDirPath = sys.argv[1]
            testingDirPath = sys.argv[2]
        else:
            trainingDirPath= '../dataset/5news-bydate-train'
            testingDirPath='../dataset/5news-bydate-test'
#             trainingDirPath= '../dataset/training'
#             testingDirPath='../dataset/testing'
        
        naivebayesui = NaiveBayesUI()
        predictionAccuracy, correctPrediction, inCorrectPrediction =\
                                    naivebayesui.naiveBayesTextClassification(\
                                                trainingDirPath, testingDirPath)

        print('\n')
        print ('------------------OUTPUT--------------------------')
        print ('correctPrediction = {} '.format(correctPrediction))
        print ('inCorrectPrediction = {} '.format(inCorrectPrediction))
        print ('predictionAccuracy = {} '.format(predictionAccuracy))
        print ('--------------------------------------------------')
