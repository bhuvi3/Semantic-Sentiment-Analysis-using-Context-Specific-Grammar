import random
lines = open('../imdb/supervised/full_data_randomized.csv').readlines()
random.shuffle(lines)
open('../imdb/supervised/full_data_randomized.csv', 'w').writelines(lines)