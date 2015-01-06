import os
#other fiels run in respective order
#to be run after updating grammars
print "###DATA ANALYSIS UPDATE STARTED###"

os.system('pyspark feature_extraction.py')

print "###FEATURE EXTRACTION DONE###"
print "###FOR BOTH WEKA AND SPARK###"

os.system('ipython data_splitter.py')

print "###DATA SPLITTING DONE###"
print "###LEARNING STARTED###"

os.system('pyspark logistic_regression_lexical_sentiment_analysis.py')

print "###LOGISTIC REGRESSION DONE###"

os.system('pyspark linear_svm_lexical_sentiment_analysis.py')

print "###LINEAR SV DONE###"
print "###DATA ANALYSIS UPDATE COMPLETE###"