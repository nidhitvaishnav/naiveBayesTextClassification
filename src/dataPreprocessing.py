
import numpy as np
import os
from myIO import MyIO
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


class DataPreprocessing:
    
# |----------------------------------------------------------------------------|
# preprocessData
# |----------------------------------------------------------------------------|
    def preprocessData(self, dirPath):
        """
        Input: dirPath
        output: completeTokenList, uniqueTokenList, nDocsInClassArr
        completeTokenList is a list which has sublist of all tokens of class 
        of given directory
        uniqueTokenList is a list which has sublist of all unique tokens of class 
        of given directory
        nDocsInClassArr is a numpy array with number of documents of each class
        
        given folder takes dirPath, walk through all the directories,
        read its files, tokenize them and return the combine tokens of all
        classes, unique token list, and number of documents in each class
        """
        #variables
        completeTokenList = []
        uniqueTokenList = []
        nDocsInClassList = []
        
        #walking through all internal directories, reading files, finding tokens
        for currentRoot,dirs,files in os.walk(dirPath):
            #finding number of files in given directory and assigning it to list 
            nFiles = len(files)
            nDocsInClassList.append(nFiles)
        
            #walking through all files in the currentDir
            classTokenList = []
            for currentFile in files:
                #finding file path of current Directory and reading its content
                currentFilePath = os.path.join(currentRoot, currentFile)
                myIO = MyIO()
                currentInputStr = myIO.readDoc(docPath = currentFilePath)
                #finding token of given file
                fileTokenList = self._tokenizationFilter(rowStr=currentInputStr)
                #adding given file token list to class token list
                classTokenList.extend(fileTokenList)
            #for currentFile -ends
            #appending classTokenList to completeTokenList
            completeTokenList.append(classTokenList)
            #finding unique class token list
            uniqueClassTokenList = list(set(classTokenList))
            uniqueTokenList.append(uniqueClassTokenList)
        #for currentRoot,dirs,files -ends
        
        #Assuming that our currentFile path is a train/test path, which contains
        #all the classDir, and no files, and the classDir contains all the file
        #Now, root directory does not provide class info. So removing its data
        completeTokenList.pop(0)
        uniqueTokenList.pop(0)
        nDocsInClassList.pop(0)
        nDocsInClassArr = np.array(nDocsInClassList)
        #returning outputs
        return completeTokenList, uniqueClassTokenList, nDocsInClassArr
#|--------------------------preprocessData -ends-------------------------------|
#|-----------------------------------------------------------------------------|
# tokenizingString
#|-----------------------------------------------------------------------------|
    def _tokenizationFilter(self, rowStr):
        """
        input: string
        output: tokenList
        given function takes an input string, 
        tokenize it with alpha - numeric characters, 
        remove digits in the string format,
        remove stop words 
        """
        #tokenizing string, with only alpha - numeric characters
        tokenizer = RegexpTokenizer(r'\w+')
        alnumTokenList = tokenizer.tokenize(rowStr)
        
        #remove degits
        alphaTokenList = []
        for currentToken in alnumTokenList:
            if(currentToken.isdigit() is False):
                alphaTokenList.append(currentToken)
            #if currentToken -ends
        #for currentToken -ends
        
        #remove stop words
        filteredTokenList = [word for word in alphaTokenList if word not in\
                                                    stopwords.words('english')]
        return filteredTokenList
    
#|------------------------tokenizingString -ends----------------------------------|    
