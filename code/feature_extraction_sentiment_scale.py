from pyspark import SparkConf, SparkContext
import re
import os
import nltk.data
import nltk
import unirest
import csv

conf = (SparkConf().setMaster("local[8]").setAppName("lexical_sentiment_feature_extraction_sentiment_scale").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

#select proper data file
review_data = sc.textFile("../imdb/supervised/full_data_randomized.csv")
#review_data = sc.textFile("../imdb/unsupervised/unsupervised_data.csv")
#review_data = sc.textFile("../imdb/train/train_data.csv")
#review_data = sc.textFile("../imdb/test/test_data.csv")

#grammar matching after sentence breaking
sentence_breaker = nltk.data.load('tokenizers/punkt/english.pickle')
def movie_review_pos_grammar_list():
	list = []
	with open("pos_grammar.txt") as fh:
		for line in fh:
			list.append(line[:-1])
	return list

def movie_review_neg_grammar_list():
	list = []
	with open("neg_grammar.txt") as fh:
		for line in fh:
			list.append(line[:-1])
	return list

def movie_review_pos_score(review):
	p_grammar = movie_review_pos_grammar_list()
	flag = 0
	sentences = sentence_breaker.tokenize(review)
	for sen in sentences:
		for grammar	in p_grammar:
			tokens = grammar.split('###')
			y = tokens[0]
			n = tokens[1]
			y_obj = re.findall(y,sen)
			n_obj = re.findall(n,sen)
			if len(y_obj) != 0:# and len(n_obj) == 0:
				#print "match review-pos found"
				flag += 1
				#break;
	return flag

def movie_review_neg_score(review):
	n_grammar = movie_review_neg_grammar_list()
	flag = 0
	sentences = sentence_breaker.tokenize(review)
	for sen in sentences:
		for grammar in n_grammar:
			tokens = grammar.split('###')
			y = tokens[0]
			n = tokens[1]
			y_obj = re.findall(y,sen)
			n_obj = re.findall(n,sen)
			if len(y_obj) != 0:# and len(n_obj) == 0:
				#print "match review-neg found"
				flag += 1
				#break;
	return flag

def get_lexical_sentiment_score_pol(review):
    pos_score = movie_review_pos_score(review)
    neg_score = movie_review_neg_score(review)
    #lexical_polarity = max(pos_score,neg_score)
    neg_score1 = neg_score
    lexical_polarity = ""
    if pos_score > neg_score:
        lexical_polarity = "pos"
    elif pos_score < neg_score:
        lexical_polarity = "neg"
    elif pos_score == neg_score:
        lexical_polarity = "nut"
    if neg_score1 == 0: #to overcome divide by zero exception
        neg_score1 = 1
    sentiment_ratio = float(pos_score)/neg_score1
    return (sentiment_ratio, lexical_polarity, pos_score, neg_score)

#returns a tuple of (pos dict, neg dict)
def get_feature_match_count_dict(review):
    sentences = sentence_breaker.tokenize(review)
    pos_grammar_match_counts = []
    neg_grammar_match_counts = []
    p_grammars = movie_review_pos_grammar_list()
    n_grammars = movie_review_neg_grammar_list()
    
    for grammar in p_grammars:
        match_count = 0
        for sen in sentences:
			tokens = grammar.split('###')
			y = tokens[0]
			n = tokens[1]
			y_obj = re.findall(y,sen)
			n_obj = re.findall(n,sen)
			if len(y_obj) != 0:# and len(n_obj) == 0:
				#print "match user-pos found"
				match_count += 1
				#break;
        pos_grammar_match_counts.append(match_count)
    
    for grammar in n_grammars:
        match_count = 0
        for sen in sentences:
			tokens = grammar.split('###')
			y = tokens[0]
			n = tokens[1]
			y_obj = re.findall(y,sen)
			n_obj = re.findall(n,sen)
			if len(y_obj) != 0:# and len(n_obj) == 0:
				#print "match user-pos found"
				match_count += 1
				#break;
        neg_grammar_match_counts.append(match_count) 
    
    #here we have match_counts for all grammars (features) from all sentences of a review (dataset)
    return (pos_grammar_match_counts, neg_grammar_match_counts)

def feature_mapper(x):
    attrib = x.split(',')
    id = attrib[0]
    rating = attrib[1]
    review = attrib[2][1:-1]
    polarity = attrib[3]
    class_polarity = str(polarity)
    score_pol = get_lexical_sentiment_score_pol(review)
    lexical_sentiment_score, lexical_polarity, pos_score, neg_score = score_pol
    #NOTE: lexical_polarity is not used as training class! Actual polarity in the data is used as training class
    #only lexical_sentiment_score is used as a complex feature.
    lex_score_complex_feature = [lexical_sentiment_score]
    
    feature_map_dict = get_feature_match_count_dict(review)
    #pos_grammar_match_count_list, neg_grammar_match_count_list = feature_map_dict
    #pos_values = pos_grammar_match_count_dict.values()
    #neg_values = neg_grammar_match_count_dict.values()
    pos_values, neg_values = feature_map_dict
    #should change class polarity to 0 and 1 instead of neg and pos respectively
    if class_polarity == 'neg':
        class_polarity = 0
    elif class_polarity == 'pos':
        class_polarity = 1
    else:
        print "***DATA EROR***"
    class_polarity = [class_polarity]
    class_rating = [rating]
    ref_id = [id]
    feature_values = ref_id + pos_values + neg_values + lex_score_complex_feature + class_polarity + class_rating
    #append other attributes that might be necessary along with data
    return tuple(feature_values)

featurized_data = review_data.map(feature_mapper)
collected_featurized_data = featurized_data.collect()

#writing to file
out = open("../imdb/supervised/featurized_lexical_analyzed_data_SENTI_SCALE.csv",'wb')
#out = open("../imdb/unsupervised/featurized_lexical_supervised_data.csv",'wb')
#out = open("../imdb/train/featurized_lexical_analyzed_train_data.csv",'wb')
#out = open("../imdb/test/featurized_lexical_analyzed_test_data.csv",'wb')

out_csv = csv.writer(out, delimiter=",")
for row in collected_featurized_data:
    #row = [str(i) for i in row]#.encode('utf-8')
    #"""
    new_row = []
    for i in row:
        if isinstance(i, (int, long, float, complex)):
            new_row.append(str(i))
        else:
            new_row.append(i.encode('utf-8'))
    row = new_row
    #"""
    out_csv.writerow(row)
out.close()

#""" getting featurized data for weka
def feature_mapper_weka(x):
    attrib = x.split(',')
    id = attrib[0]
    rating = attrib[1]
    review = attrib[2][1:-1]
    polarity = attrib[3]
    class_polarity = str(polarity)
    score_pol = get_lexical_sentiment_score_pol(review)
    lexical_sentiment_score, lexical_polarity, pos_score, neg_score = score_pol
    #NOTE: lexical_polarity is not used as training class! Actual polarity in the data is used as training class
    #only lexical_sentiment_score is used as a complex feature.
    lex_score_complex_feature = [lexical_sentiment_score]
    
    feature_map_dict = get_feature_match_count_dict(review)
    #pos_grammar_match_count_list, neg_grammar_match_count_list = feature_map_dict
    #pos_values = pos_grammar_match_count_dict.values()
    #neg_values = neg_grammar_match_count_dict.values()
    pos_values, neg_values = feature_map_dict
    #should keep class polarity as neg and pos for WEKA
    class_polarity = [class_polarity]
    class_rating = [rating]
    ref_id = [id]
    feature_values = ref_id + pos_values + neg_values + lex_score_complex_feature + class_polarity + class_rating
    #append other attributes that might be necessary along with data
    return tuple(feature_values)
    
featurized_data_weka = review_data.map(feature_mapper_weka)
collected_featurized_data_weka = featurized_data_weka.collect()

#writing to file
out = open("../imdb/supervised/featurized_lexical_analyzed_data_SENTI_SCALE_weka.csv",'wb')
#out = open("../imdb/unsupervised/featurized_lexical_supervised_data.csv",'wb')
#out = open("../imdb/train/featurized_lexical_analyzed_train_data.csv",'wb')
#out = open("../imdb/test/featurized_lexical_analyzed_test_data.csv",'wb')

out_csv = csv.writer(out, delimiter=",")
for row in collected_featurized_data_weka:
    #row = [str(i) for i in row]#.encode('utf-8')
    #"""
    new_row = []
    for i in row:
        if isinstance(i, (int, long, float, complex)):
            new_row.append(str(i))
        else:
            new_row.append(i.encode('utf-8'))
    row = new_row
    #"""
    out_csv.writerow(row)
out.close()
#"""