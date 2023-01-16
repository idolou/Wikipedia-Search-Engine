import math
import nltk
import re
import pickle
from inverted_index_gcp import *
import numpy as np
from gensim.models import KeyedVectors
import heapq
# from numpy.linalg import norm
import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')
# from nltk import PorterStemmer


def read_pickle(file_name):
  """
  function to read pickle file from google storage bucket.
  
  Parameters:
  -----------
  file_name: the name of the pickle file to read
  
  Returns:
  -----------
  the pickle file
  """
  with open(f"postings_gcp/{file_name}.pkl", 'rb') as f:
    return pickle.loads(f.read())


# save title to store original docs title for retrival
id_title_dict = read_pickle("id_title_dict")

#save doc length dictioneries for body and title
#will use it in the bm25 score calculation
DL_body = read_pickle("DL_body")
DL_title = read_pickle("DL_title")

# save model for word2vec query expantion
model = KeyedVectors.load_word2vec_format("postings_gcp/model_wiki.bin", binary = True)



def expand_query(tokens, size):
  
  add_tokens = []
  if size>3 or size == 1:
    try:
      new_tokens = list(map(lambda x: x[0].lower(), model.most_similar(positive=tokens, topn=8)))
      filtered_tokens = []
      for word in new_tokens:
        for token in tokens:
          if (word.find(token) != -1 or token.find(word) != -1):
            filtered_tokens.append(word)
      add_tokens += filtered_tokens

    except:
      pass
  tokens+=add_tokens
  return set(tokens)



english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"] ###check

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)

def tokenize(text):
  """ Tokenize text into words, removing punctuation and stopwords.
  Parameters:
  -----------
  text: string

  Returns:
  -----------
  tokens: list of strings
  """
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  tokens = [token for token in tokens if token not in all_stopwords]
  return tokens




def get_top_n(sim_dict,N=100):
    """ 
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores 
   
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3
    
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return heapq.nlargest(N, [(doc_id,score) for doc_id, score in sim_dict.items()], key = lambda x: x[1])

def get_title_by_doc_id(doc_id):
    """
    function to get the title of a document by its id.

    Parameters:
    -----------
    doc_id: the id of the document
    title_doc_dict: a dictionary of document id and title


    Returns:
    -----------
    the title of the document
    """
    return id_title_dict.get(doc_id, "Bad Title!")


def search_body_tfidf_cosine(query_to_search, index, N, kind=""):
  """
  Search for documents based on the query using TF-IDF and cosine similarity.
  The query will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
  The function will return a ranked list of documents based on the cosine similarity score.
  For calculation of IDF, use log with base 10.
  tf will be normalized based on the length of the document.

  Parameters:
  -----------
  query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
  index: inverted index
  N: Integer (how many documents to retrieve). By default N = 3
  kind: string (the type of the index). By default kind = ""

  Returns:
  -----------
  a ranked list of pairs (doc_id, score) in the length of N.
  """
  DL = read_pickle("Cosine_norm_dict")
  words = index.df.keys()
  normalized_tfidf_docs = defaultdict(list)
  query_tfidf = {}
  cos_sin_dict = defaultdict()
  query = set(query_to_search)
  counter = Counter(query_to_search)
  epsilon = .0000001


  for term in query:
    if term in words:

      candidates = index.search(term, kind)
      tf = counter[term]/len(query_to_search) # term frequency divded by the length of the query
      df = index.df[term]
      idf = math.log((index.N)/(df+epsilon),10) #smoothing
      query_tfidf[term] = tf * idf

      for doc_id, tf in candidates:
        ### check
        if doc_id in DL:
          tfidf = (doc_id,(tf/DL_body[doc_id])*math.log(len(DL_body)/index.df[term],10))
          normalized_tfidf_docs[term].append(tfidf)

      for token, candidate in normalized_tfidf_docs.items():
        for doc_id, tfidf in candidate:
          numerator = tfidf * query_tfidf[token]
          denominator = DL[doc_id] * math.sqrt(sum([x**2 for x in query_tfidf.values()]))
          if denominator > 0:
            cos_sin_dict[doc_id] = numerator/denominator


  top_n_docs = get_top_n(cos_sin_dict, N)

  topN_id = [tup[0] for tup in top_n_docs]

  return [(id, get_title_by_doc_id(id)) for id in topN_id]





def doc_binary_search(query_to_search, index, kind=""):
  """
  Help function 

  Parameters:
  -----------
  query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.'). 
                    Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

  index_title:     inverted index loaded from the corresponding files.
  N:               Integer. How many documents to retrieve. By default N = 100.

  Returns:
  -----------
  return: a list of pairs in the following format:(doc_id, title). 
  """

  candidates = defaultdict(int)
  for term in set(query_to_search):
      if term in index.df.keys():

        docs = index.search(term, kind)
        for doc_id, freq in docs:
          candidates[doc_id] += 1


  candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
  return candidates





def generate_binary_result(query_to_search, index, kind=""):
  """
  Help function

  Parameters:
  -----------
  topDocs_list: list of pairs in the following format:(doc_id, score).
  title_dict:   dictionary of titles as follows:
                                          key: doc_id
                                          value: title

  Returns:
  -----------
  return: a list of pairs in the following format:(doc_id, title).
  """
  docs = doc_binary_search(query_to_search, index, kind)
  return [(doc_id, get_title_by_doc_id(doc_id)) for doc_id, freq in docs]



##########################################################
########-------------BM25--------------###################

class BM25_from_index:
    """
    Best Match 25.    
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self,index,k1=1.5, b=0.75, delta = 1.2):
        self.b = b
        self.k1 = k1
        self.index = index
        self.delta = delta
        self.AVGDL = index.AVGDL
        self.N = 6348910    

    def calc_idf(self,list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        
        Returns:
        -----------
        idf: dictionary of idf scores. As follows: 
                                                    key: term
                                                    value: bm25 idf score
        """        
        idf = {} 
        words = self.index.df.keys()       
        for term in set(list_of_tokens):            
            if term in words:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass                             
        return idf
        


    def calc_score(self, query, N, kind=""):
      """
      This function calculate the bm25 score for given query and document.
      We need to check only documents which are 'candidates' for a given query.
      This function return a dictionary of scores as the following:
                                                                  key: query_id
                                                                  value: a ranked list of pairs (doc_id, score) in the length of N.

      Parameters:
      -----------
      query: list of token representing the query. For example: ['look', 'blue', 'sky']
      doc_id: integer, document id.

      Returns:
      -----------
      dict of scores: float, bm25 score.
      """
      self.idf = self.calc_idf(query)
      candidates = defaultdict(int)
      if kind == "body_index":
        DL = DL_body
      elif kind == "title_index":
        DL = DL_title

      for term in set(query):
        if term in self.index.df.keys():
          posting_list = self.index.search(term, kind)
          for doc_id, tf in posting_list:
            numerator = tf * (self.k1 + 1)
            denominator =   tf + self.k1 * (1 - self.b + self.b * DL[doc_id] / self.AVGDL)
            candidates[doc_id] +=  self.idf[term] * ((numerator / denominator)+self.delta)

      sort_top_n = heapq.nlargest(N, [(doc_id,score) for doc_id, score in candidates.items()], key = lambda x: x[1])
      return sort_top_n






def get_BM25_score(query, index, N, k, b, kind=""):

  bm25_index = BM25_from_index(index, k1=k, b=b)
  return bm25_index.calc_score(query, N, kind=kind)



def merge_score_results(query, body_index, title_index, body_w = 0.33, 
  title_w = 0.67, N = 100, title_query_similarity = False):
  """
    This function merge the results of the different models.
    The results are merged according to the following formula:
    score = body_w * body_score + title_w * title_score + anchor_w * anchor_score + pr_w * pr_score + pv_w * pv_score
    
    Parameters:
    -----------
    body_res: list of pairs (doc_id, score) in the length of N.
    title_res: list of pairs (doc_id, score) in the length of N.

    body_w: float, weight of the body score.
    title_w: float, weight of the title score.

    pr_w: float, weight of the page rank score.
    pv_w: float, weight of the page view score.
    
    Returns:
    -----------
    list of pairs (doc_id, score) in the length of N
    """
  
  query_tokenized = tokenize(query)
  extanded_query = expand_query(query_tokenized, len(query.split(" ")))
  # title_query = [word for word in query_tokenized if word not in ['make']]
  merged_docs = defaultdict(float)
  window = 200
  body_res = get_BM25_score(extanded_query, body_index, N=window, k=1.5, b =0.2, kind="body_index")

  title_res = get_BM25_score(query_tokenized, title_index, N=window, k=1.4, b =0.7, kind="title_index")

  
  for doc_id, score in title_res:
    merged_docs[doc_id] += title_w * score
  for doc_id, score in body_res:
    merged_docs[doc_id] +=  score *body_w


  boost = 1.5
  if title_query_similarity:
    for doc_id, score in merged_docs.items():
      title = get_title_by_doc_id(doc_id)
      merged_docs[doc_id] += similarity(query_tokenized, title) * boost


  merged_best = heapq.nlargest(N, [(doc_id,score) for doc_id, score in merged_docs.items()], key = lambda x: x[1])


  return [(tup[0], get_title_by_doc_id(tup[0])) for tup in merged_best]



def similarity(query, title):
  """
  This function calculate the similarity between the query and the title.
  The similarity is calculated according to the following formula:
  similarity = sum of the similarity between each term in the query and the title / number of terms in the query

  Parameters:
  -----------
  query: list of token representing the query. For example: ['look', 'blue', 'sky']
  title: string, title of the document.

  Returns:
  -----------
  float, similarity score.
  """
  stemmer = PorterStemmer()
  similiarities = []
  title_tokenized = tokenize(title)
  for token in query:
    for term in title_tokenized:
      stem_token = stemmer.stem(token)
      stem_term = stemmer.stem(term)
      if all(x in model.wv.vocab for x in [stem_token,stem_term]):
        similiarities.append(model.similarity(stem_token, stem_term))
  if len(similiarities) == 0:
    return 0
  return sum(similiarities)/len(similiarities)













    
    