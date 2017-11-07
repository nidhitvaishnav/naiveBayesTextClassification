
1. We have used Python to implement MultiNomial Naive Bayes Algorithm for Text Classification.
2. The code is written in Eclipse IDE. The project name is NaiveBayes. It comprises src folder which contains
   following .py files.
1) naiveBayesUI.py
2) dataPreprocessing.py
3) myIO.py
4) TrainMultinomialNaiveBayesIO.py

3. naiveBayesUI.py can be run to run the entire code. Following two arguments should be provided as 
   command line arguments:
   1) Path of Training Dataset Folder
   2) Path of Testing Dataset Folder
   Training Dataset folder should consist of subfolders belonging to different classes which contains
   documents related to that class. The code run for any number of classes included in the training and 
   testing dataset paths.

4. From all documents of each class, the data is read and tokenized into words using nltk library. 
   Stop words are removed using nltk. All tokens are converted to lower case so case sensitive behavior 
   is not affected in performance. Numerical data are also cleaned from the data. Special 
   character inbetween any two words is considered as delimeter and are tokenized. e.g. R2-D2 word is
   splitted into R2 and D2.

   install nltk Corpora - stopwords
   install nltk Models punkt
   go to python command prompt
   >>import nltk
   >>nltk.download()
   go to Models, find punkt in identifiers and click download 
   Now go to Corpora, find stopwords in identifiers and click download

   
5. TrainMultinomialNaiveBayesIO.py trains the model for the training dataset and calculates proir
   probability of each class and conditional probability of each term for each class. Then testing 
   is performed by predicting the class having maximum score probability. Accuracy is shown as an
   output.


6. There are some words which are not found anywhere in the training dataset but are encountered in the 
   testing dataset. For such words, conditional probability is not found so they are ignored.

7. The model takes few minutes for preprocessing of data and showing the results. So progress is shown
   as an output continuously. Then the accuracy in % is displayed.

8. The screenshots included in the directory shows the output.



Analysis of Results:


1) If classes of same domain are trained and accuracy is calculated based on them, it is observed that
   accuracy obtained will be low comparatively. The reason for this is that if the classes are under same
   domain, they will have many terms in common. This will misclassify many documents.

2) If the classes of different domains are trained and accuracy is measured, accuracy will be quite high.
   The terms in these documents will be quite different and probability of misclassification reduces.

