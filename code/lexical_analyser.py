from pyspark import SparkConf, SparkContext
import re
import os
import nltk.data
import nltk
import unirest
import csv

conf = (SparkConf().setMaster("local[8]").setAppName("bank_marketing_classification_linear_svm").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

review_data = sc.textFile("../imdb/unsupervised/unsupervised_data.csv")
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

def lexical_analyser(x):
    attrib = x.split(',')
    id = attrib[0]
    rating = attrib[1]
    review = attrib[2][1:-1]
    polarity = attrib[3]
    score_pol = get_lexical_sentiment_score_pol(review)
    lexical_sentiment_score, lexical_polarity, pos_score, neg_score = score_pol
    return (id, rating, review, polarity, lexical_polarity, lexical_sentiment_score, pos_score, neg_score)


lexical_supervised_data = review_data.map(lexical_analyser)
sup_data = lexical_supervised_data.collect()

#writing to file
out = open("../imdb/unsupervised/lexical_supervised_data.csv",'wb')
#out = open("../imdb/train/lexical_analyzed_train_data.csv",'wb')
#out = open("../imdb/test/lexical_analyzed_test_data.csv",'wb')

out_csv = csv.writer(out, delimiter=",")
#print sup_data[0]
for row in sup_data:
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