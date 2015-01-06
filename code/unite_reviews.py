import os
from os import listdir
import csv
import re

#dir = "../imdb/unsupervised/"
#dir = "../imdb/train/"
dir = "../imdb/test/"
os.chdir(dir)
#out = open("unsupervised_data.csv","wb")
#out = open("train_data.csv","wb")
out = open("test_data.csv","wb")
c_out = csv.writer(out, delimiter=",")
#id,rating,review,polarity, (sentiment_polarity)

"""
unsup_files = os.listdir("./unsup")
pol = "u"
i = 1
for f in unsup_files:
    tok = f.split('.')[0]
    tok = tok.split('_')
    #id = int(tok[0]) + 1
    #id = tok[0]
    id = i
    i = i + 1
    rating = tok[1]
    fh = open("./unsup/"+f)
    review = fh.readline()
    review =  re.sub('<br /><br />', ' ', review)
    #review =  re.sub("[^A-Za-z0-9 ';:$&%@!]+", '', review)
    review =  re.sub('[,"]', '', review)
    review = ' '.join(review.split())
    review = '|' + review + '|'
    fh.close()
    row = []
    row.extend([id,rating,review,pol])
    c_out.writerow(row)
"""
#"""
p_files = os.listdir("./pos")
pol = "pos"
i = 1
for f in p_files:
    tok = f.split('.')[0]
    tok = tok.split('_')
    id = i
    i = i + 1
    rating = tok[1]
    fh = open("./pos/"+f)
    review = fh.readline()
    review =  re.sub('<br /><br />', ' ', review)
    #review =  re.sub("[^A-Za-z0-9 ';:$&%@!]+", '', review)
    review =  re.sub('[,"]', '', review)
    review = ' '.join(review.split())
    review = '|' + review + '|'
    fh.close()
    row = []
    row.extend([id,rating,review,pol])
    c_out.writerow(row)

#"""
#"""	
n_files = os.listdir("./neg")
pol = "neg"
for f in n_files:
    tok = f.split('.')[0]
    tok = tok.split('_')
    id = int(tok[0]) + 12501
    id = i
    i = i + 1
    rating = tok[1]
    fh = open("./neg/"+f)
    review = fh.readline()
    review =  re.sub('<br /><br />', ' ', review)
    #review =  re.sub("[^A-Za-z0-9 ';:$&%@!]+", '', review)
    review =  re.sub('[,"]', '', review)
    review = ' '.join(review.split())
    review = '|' + review + '|'
    fh.close()
    row = []
    row.extend([id,rating,review,pol])
    c_out.writerow(row)

out.close()
#"""
#