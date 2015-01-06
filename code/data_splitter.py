import os

#"""
#Breaking Classification Data for Sentiment Polarity
full_fh = open('../imdb/supervised/featurized_lexical_analyzed_data.csv')
full_lines = full_fh.readlines()
train_fh = open('../imdb/supervised/featurized_train_data.csv','w')
test_fh = open('../imdb/supervised/featurized_test_data.csv','w')

train_lines = full_lines[:45000]
test_lines = full_lines[-5000:]

train_fh.writelines(train_lines)
test_fh.writelines(test_lines)

train_fh.close()
test_fh.close()
full_fh.close()
#"""

"""
#Breaking Linear Regression Data for Sentiment Scale
full_fh = open('../imdb/supervised/featurized_lexical_analyzed_data_SENTI_SCALE.csv')
full_lines = full_fh.readlines()
train_fh = open('../imdb/supervised/featurized_train_data_SENTI_SCALE.csv','w')
test_fh = open('../imdb/supervised/featurized_test_data_SENTI_SCALE.csv','w')

train_lines = full_lines[:45000]
test_lines = full_lines[-5000:]

train_fh.writelines(train_lines)
test_fh.writelines(test_lines)

train_fh.close()
test_fh.close()
full_fh.close()
#"""
