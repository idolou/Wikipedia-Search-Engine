# Search Engine Wikipedia
This project is part of the Information Retrieval Course. We built the inverted indexes using GCP cluster and pyspark RDDs, the engine works on the english wikipedia dump from 2021 which  includes over 6,000,000 million documents. the Search Engine runs on GCP VM Linux Instance.

## The project includes 3 main python files:
#### 1. search_frontend.py:
  this python script run the flask application that allows users to send request via url and get the results (wikipedia document id's and titles).
#### 2. search_backend.py:
  this python script run all the logic behind the search Engine.
#### 3. inverted_index_gcp.py:
  this file includes the class of Inverted_Index which save the indexs of all english wikipedia document for body, title and anchor texts.
  
  
## Search Engine functionality 
The search engine can preform variant kind of searches on wkikpedia corpus
### Search - 
Search over Wikipedia documents. The documents is ranked by BM25 algorithm both for body text and title text. the engine do query expantion to get similar words to search for by word2vec model.
### Search_body - 
Search over Wikipedia only by the text of the documents body. The documents relevancy is ranked by Cosine-Similarity score using TF-IDF.
### Search_title - 
Search over Wikipedia only by the titles of the documents. The documents relevancy is ranked by binary search.
### Search_anchor - 
Search over Wikipedia only by the anchor text related to the documents. The documents relevancy is ranked by binary search as well.
### get_pagerank - 
Search for the page rank of specific documents, the calculation of the page rank was precalculated by pyspark graphframes.
### get_pageview - 
Search for the page view of specific documents, the data was online and loades as python dictionary.

# Exemples
snapshot of top results for the query Information Retrieval:

![image](https://user-images.githubusercontent.com/63515984/212550462-8d49bc73-3481-43d5-afb2-41395798923b.png)


query expantion:

![image](https://user-images.githubusercontent.com/63515984/212551063-54016bc0-0bdc-4a90-a80c-b686ea8794b1.png)

similarity between title and query:

![image](https://user-images.githubusercontent.com/63515984/212551117-32389fd1-5920-4858-82bb-e5b65ac63d61.png)

