import io

class MyIO:
    
# |----------------------------------------------------------------------------|
# readDoc
# |----------------------------------------------------------------------------|
    def readDoc(self, docPath):
        """
        given function reads document and removes unnecessary headers
        """
#         docFile = open(docPath, "r");
        docFile = io.open(docPath, 'r', encoding='ISO-8859-1')
#         docFile = io.open(docPath, 'r', encoding='utf8')
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
    
#|--------------------------readDoc -ends--------------------------------------|