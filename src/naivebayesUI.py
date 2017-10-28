import sys
import os
import numpy as np
from dataPreprocessing import DataPreprocessing

class NaiveBayesUI:
    
    def preprocessData(self, trainingFolderPath, testingFolderPath):
        '''
            This function takes training folder path and testing folder path as arguments and
            
            
        '''
        dataPreprocessing=DataPreprocessing()
        dataPreprocessing.preprocessData()                
               
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
        
        ''' This function takes two command line arguments - one is the path of the folder containing training data and 
            the other is the path of the folder containing testing data'''
        
        if len(sys.argv)>1:
            trainingFolderPath = sys.argv[1]
            testingFolderPath = sys.argv[2]
        else:
            trainingFolderPath= '../dataset/5news-bydate-train'
            testingFolderPath='../dataset/5news-bydate-test'
        
        naivebayesui = NaiveBayesUI()
        naivebayesui.preprocessData(trainingFolderPath, testingFolderPath)
        
        
        
    