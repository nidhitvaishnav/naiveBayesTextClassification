
import numpy as np
import os

class DataPreprocessing:
    
# |---------------------------------------------------------------|
# preprocessData
# |---------------------------------------------------------------|
    def preprocessData(self, folderPath):
        """
        given folder takes training and testing data and perform preprocessing
        task
        Input: folderpath
        output: 
        """
        docsInClassArr = np.array([])
        for currentRoot,dirs,files in os.walk(folderPath):
            print("root:\n{}\n".format(currentRoot))
            print("dirs: \n{}\n".format(dirs))
            print("files:\n{}\n".format(files))
            
            nFiles = len(files)
            nDirs = len(dirs)
            docsInClassArr=np.append(docsInClassArr, nFiles)
#             #debug
#             print("docsInClassArr : {}".format(docsInClassArr))
#             #debug -ends

            for currentFile in files:
                currentFilePath = os.path.join(currentRoot, currentFile)
                #debug
                print("currentFilePath : {}".format(currentFilePath))
                #debug -ends
                
#             for classk in dirs:
#                 className=classk
#                 print(className)
#                 numberofDocsOfThisClass=0
#                 for filek in files:
#                     print(filek)
#                     numberOfDocs+=1
#                     numberofDocsOfThisClass+=1
#                     print(numberofDocsOfThisClass)

    
#|--------------------------preprocessData -ends-------------------|


if __name__ == '__main__':
    dataPreprocessing = DataPreprocessing()
    dataPreprocessing.preprocessData('../dataset/5news-bydate-train')
    