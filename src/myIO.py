
class MyIO:
    
# |---------------------------------------------------------------|
# readDoc
# |---------------------------------------------------------------|
    def readDoc(self, docPath):
        """
        given function reads document and removes unnecessary headers
        """
        docFile = open(docPath, "r");
        outputStr = ""
        writeFlag = False   
        for myLine in (docFile):
#             #debug
#             print("line : {}".format(myLine[0]))
#             #debug -ends 
            if (writeFlag):
            
                outputStr = outputStr+" "+myLine
            #if writeFlag -ends
            
            if(myLine.split(' ')[0]=='Lines:'):
                writeFlag=True
            #if myLine -ends
        #for myLine -ends
        return outputStr
#|--------------------------readDoc -ends-------------------------|


if __name__ == '__main__':
    myIO = MyIO()
    outputStr =myIO.readDoc('../dataset/5news-bydate-train/comp.windows.x/67391')
    #debug
    print("outputStr : {}".format(outputStr))
    #debug -ends