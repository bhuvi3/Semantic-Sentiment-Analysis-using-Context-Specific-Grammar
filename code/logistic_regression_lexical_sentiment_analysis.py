from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array
import math
import time

#building Spark Configuration and Getiing a Spark Context and Loading Data into an RDD
conf = (SparkConf().setMaster("local[8]").setAppName("lexical_sentiment_analysis_logistic_regression").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

training_data = sc.textFile('../imdb/supervised/featurized_train_data.csv')
testing_data = sc.textFile('../imdb/supervised/featurized_test_data.csv')

#training_data = sc.textFile("../imdb/featurized_lexical_analyzed_supervised_data_training_set.csv")
#testing_data = sc.textFile("../imdb/featurized_lexical_analyzed_supervised_data_testing_set.csv")

#parsing mapper function
def mapper_CF(x):
    z = str(x)
    tokens = z.split(',')
    c = len(tokens)
    total_attrib = 12
    class_index = 11
    feature_index_start, feature_index_end = (0, 10) #included
    if c != total_attrib:
        print "*********DATA VALIDATION ERROR\n************"
    attrib = [float(y) for y in tokens]
    #attrib[11] = int(attrib[11])
    return LabeledPoint(attrib[class_index], attrib[feature_index_start:(feature_index_end+1)])

vectorize_start = time.time()
vectorized_data = training_data.map(mapper_CF)
vectorized_testing_data = testing_data.map(mapper_CF)
"""
train_instances = vectorized_data.count()
test_instances = vectorized_testing_data.count()
total_instances = train_instances + test_instances
train_per = float(train_instances)/total_instances * 100
test_per = float(test_instances)/total_instances * 100
#"""
vectorize_end = time.time()
print "******************VECTORIZING: DONE********************"

#building a logistic regression training model
train_start = time.time()
model = LogisticRegressionWithSGD.train(vectorized_data)
train_end = time.time()
print "******************MODEL TRAINING: DONE********************"

#predicting classes for testing data and evaluating
def mapper_predict(x):
    predicted_class = model.predict(x.features)
    #predicted_class = int(round(predicted_class))
    actual_class = x.label
    return (actual_class, predicted_class)

pred_start = time.time()
actual_and_predicted = vectorized_testing_data.map(mapper_predict)
count = actual_and_predicted.count()
pred_end = time.time()
print "******************PREDICTION: DONE********************"

#evaluation
eval_start = time.time()
training_error = actual_and_predicted.filter(lambda (a, p): a != p).count() / float(count)
MSE = actual_and_predicted.map(lambda (a, p): (a - p)**2).reduce(lambda x, y: x + y) / count
RMSE = math.sqrt(MSE)
eval_end = time.time()

#confusion matrix: classes: pos(1), neg(0)
accuracy = (1 - training_error) * 100
p_actual_p_predicted = actual_and_predicted.filter(lambda (a, p): a == 1 and p == 1).count()
p_actual_n_predicted = actual_and_predicted.filter(lambda (a, p): a == 1 and p == 0).count()
n_actual_p_predicted = actual_and_predicted.filter(lambda (a, p): a == 0 and p == 1).count()
n_actual_n_predicted = actual_and_predicted.filter(lambda (a, p): a == 0 and p == 0).count()
print "******************EVALUATION: DONE********************"

#efficiency: time calculation
vectorize_time = vectorize_end - vectorize_start
train_time = train_end - train_start
pred_time = pred_end - pred_start
eval_time = eval_end - eval_start
print "******************TIME CALCULATION: DONE********************\n"

print "******************RESULTS********************"
heading = "***LEXICAL SENTIMENT ANALYSIS***\n"
#"""
data_title = "Large Movie Review Dataset v1.0 - IMDB" + "\n"
instances = "Total: " + str(50000) + "\n"
instances += "Train: " + str(45000) + "=>" + str(90) + "%" + "\n"
instances += "Test: " + str(5000) + "=>" + str(10) + "%" + "\n"
data_info = data_title + instances
#"""
"""
data_title = "Large Movie Review Dataset v1.0 - IMDB" + "\n"
instances = "Total: " + str(total_instances) + "\n"
instances += "Train: " + str(train_instances) + "=>" + str(train_per) + "\n"
instances += "Test: " + str(test_instances) + "=>" + str(test_per) + "\n"
data_info = data_title + instances
#"""
title = "\n***LOGISTIC REGRESSION CLASSIFIER RESULTS***\n"

accuracy_title = "\n##############Accuracy##################\n\n"
accuracy_line = ("Accuracy = " + str(accuracy) + "%") + '\n'
train_err_res = ("Training Error = " + str(training_error)) + '\n'
rmse_res = ("Mean Squared Error = " + str(RMSE)) + '\n'

efficiency = "\n##############Efficiency################\n\n"
vectorize_res = ("Vectorizing Time = " + str(vectorize_time)) + '\n'
train_res = ("Training TIme = " + str(train_time)) + '\n'
pred_res = ("Predicting TIme = " + str(pred_time)) + '\n'
eval_res = ("Evaluation TIme = " + str(eval_time)) + '\n'

conf_title = "\n***CONFUSION MATRIX***\n"
row0 = str(accuracy) + "%" + "\t|\t" + "pos" + "\t|\t" + "neg" + "\n"
row_sep = "------------------------------------" + "\n"
row1 = "pos" + "\t|\t" + str(p_actual_p_predicted) + "\t|\t" + str(p_actual_n_predicted) + "\n"
row2 = "neg" + "\t|\t" + str(n_actual_p_predicted) + "\t|\t" + str(n_actual_n_predicted) + "\n"
confusion_mat = row0 + row_sep + row1 + row2
result = heading + data_info + title + accuracy_title + accuracy_line + train_err_res + rmse_res + conf_title + confusion_mat + efficiency + vectorize_res + train_res + pred_res + eval_res 

res_fh = open('../result/logistic_regression_result.txt','w')
res_fh.write(result)
res_fh.close()

print heading + title
print "\n##############ACCURACY##################"
print("Training Error (Mean Absolute Error) = " + str(training_error))
print("Root Mean Squared Error = " + str(RMSE))
print("Accuracy = " + str(accuracy) + "%")
print(conf_title + str(confusion_mat))
print "########################################"
print "\n##############EFFICIENCY################"
print("Vectorizing Time = " + str(vectorize_time))
print("Training TIme = " + str(train_time))
print("Predicting TIme = " + str(pred_time))
print("Evaluation TIme = " + str(eval_time))
print "########################################"
