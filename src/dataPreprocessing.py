
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
    def preprocessTrainingData(self, dirPath):
        """
        Input: dirPath
        output: classTokenList, uniqueTokenList, nDocsInClassArr, dirNameList
        classTokenList is a list which has sublist of all tokens of classes 
        of given directory
        uniqueTokenList is a list which has all unique tokens of all classes 
        of given directory
        nDocsInClassArr is a numpy array with number of documents of each class
        dirNameList provides all directory names (which are class names here),
        from current dirPath (training or testing)
        
        given folder takes dirPath, walk through all the directories,
        read its files, tokenize them and return the combine tokens of all
        classes, unique token list, and number of documents in each class
        """
        #variables
        classTokenList = []
        generalTokenList = []
        nDocsInClassList = []
        dirNameList = next(os.walk(dirPath))[1]
#         #debug
#         print("dirNameList : {}".format(dirNameList))
#         #debug -ends
        #walking through all internal directories, reading files, finding tokens
        for currentRoot,dirs,files in os.walk(dirPath):
            #finding number of files in given directory and assigning it to list 
            nFiles = len(files)
            nDocsInClassList.append(nFiles)
        
            #walking through all files in the currentDir
            currentClassTokenList = []
            for currentFile in files:
                #finding file path of current Directory and reading its content
                currentFilePath = os.path.join(currentRoot, currentFile)
                myIO = MyIO()
                currentInputStr = myIO.readDoc(docPath = currentFilePath)
#                 #debug
#                 print("currentInputStr : {}".format(currentInputStr))
#                 #debug -ends
                #finding token of given file
                fileTokenList = self._tokenizationFilter(rowStr=currentInputStr)
                #adding given file token list to class token list
                currentClassTokenList.extend(fileTokenList)
                generalTokenList.extend(fileTokenList)
            #for currentFile -ends
            #appending currentClassTokenList to classTokenList
            classTokenList.append(currentClassTokenList)
            
#             #putting all tokens in one token list
#             generalTokenList.extend(classTokenList)
        #for currentRoot,dirs,files -ends
#         #debug
#         print("generalTokenList : {}".format(generalTokenList))
#         #debug -ends
        uniqueTokenList = list(set(generalTokenList))
        #Assuming that our currentFile path is a train/test path, which contains
        #all the classDir, and no files, and the classDir contains all the file
        #Now, root directory does not provide class info. So removing its data
        classTokenList.pop(0)
#         uniqueTokenList.pop(0)
        nDocsInClassList.pop(0)
        nDocsInClassArr = np.array(nDocsInClassList)
        #returning outputs
        return classTokenList,uniqueTokenList, nDocsInClassArr, dirNameList
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
    
#|------------------------tokenizingString -ends-------------------------------|    
#|-----------------------------------------------------------------------------|
# preprocessTestingFile
#|-----------------------------------------------------------------------------|
    def preprocessTestingData(self, testingDirPath):
        """
        input: testingDirPath
        
        Given function walks through all files in testingDir Path, and provides
        tokens of eachFile
        """
        fileTokenDict = {}
        for currentRoot,dirs,files in os.walk(testingDirPath):
        
            #walking through all files in the currentDir
            for currentFile in files:
                #finding file path of current Directory and reading its content
                currentFilePath = os.path.join(currentRoot, currentFile)
                myIO = MyIO()
                currentInputStr = myIO.readDoc(docPath = currentFilePath)
#                 #debug
#                 print("currentInputStr : {}".format(currentInputStr))
#                 #debug -ends
                #finding token of given file
                fileTokenList = self._tokenizationFilter(rowStr=currentInputStr)
                #adding given file token list to class token list
                #fileTokenDict[currentFile] = list(set(fileTokenList))
                #fileTokenDict[currentFile] = fileTokenList
                
            #for currentFile -ends
        #for currentRoot, dirs, files -ends
        return fileTokenList,currentFile
#|------------------------preprocessTestingFile -ends--------------------------|    
