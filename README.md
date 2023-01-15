# Search Engine Wikipedia
This project is part of the Information Retrieval Course.

## The project includes 3 main python files:
### 1. search_frontend.py:
  this python script run the flask application that allows users to send request via url and get the results (wikipedia document id's and titles).
2. search_backend.py:
  this python script run all the logic behind the search Engine.
3. inverted_index_gcp.py:
  this file includes the class of Inverted_Index which save the indexs of all english wikipedia document for body, title and anchor texts.
  
  

This is a search engine for English Wikipedia.
The search engine based on veriaty of measurements and we can get the following results:
## Search - 
Search over Wikipedia. The documents relevancy is ranked by BM25 retrieval function and Pagerank.
## Search_body - 
Search over Wikipedia only by the body of the documents. The documents relevancy is ranked by Cosine-Similarity score using TF-IDF.
## Search_title - 
Search over Wikipedia only by the titles of the documents. The documents relevancy is ranked by binary search.
## Search_anchor - 
Search over Wikipedia only by the anchor text related to the documents. The documents relevancy is ranked by binary search.

![Screenshot 2022-08-04 120357](https://user-images.githubusercontent.com/63515984/182808618-a511d75a-f1ef-4237-9e2a-ec2d5a48e8de.jpg)
