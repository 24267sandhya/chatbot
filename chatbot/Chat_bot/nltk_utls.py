import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
     return nltk.word_tokenize(sentence)

def stem(word):
     return stemmer.stem(word.lower())


## matching with token words which are already stored to all words

def bag_of_words(token_sent,all_words):
     token_sent = [stem(w) for w in token_sent]             # token words simplified
     bag = np.zeros(len(all_words), dtype= np.float32)      # intiallised all to zeros
     for idx, w in enumerate(all_words):                    # addressing natural no to words in sentence
          if w in token_sent:                               # since all are intiallised to zero
               bag[idx] = 1.0                               # required token are intiallaed with 1 value
          
     return bag



     





