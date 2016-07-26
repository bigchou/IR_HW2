import os
import operator
import numpy as np
from math import log
from scipy import spatial

doc_path = 'SPLIT_DOC_WDID_NEW/'
qry_path = 'QUERY_WDID_NEW/'

filenames = [] # record each name of document
documents = [] # detail of each document
terms = [] # all terms
# Read collections of documents
for filename in os.listdir(doc_path):
    if filename != ".DS_Store":
        filenames.append(filename)
        # Get detail of each document
        with open( doc_path+filename , 'r') as fin:
            #skip the first three lines
            for i in range(3):
                fin.readline()
            # get each document
            doc = ""
            while True:
                content = fin.readline()
                if content == "":
                    break
                else:
                    for word in content.split():
                        if word != "-1":
                            doc += (word+' ')
            documents.append(doc)
            # collect all terms
            for term in doc.split():
                terms.append(term)
num_docs = len(documents) # number of documents
print('FINISH READING DOCUMENTS')

# Read collections of queries
querynames = [] # name of querys' file
querys = [] # detai of each query
for filename in os.listdir(qry_path):
    if filename != ".DS_Store":
        querynames.append(filename)
        # Get detail of each document
        with open( qry_path+filename , 'r') as fin:
            doc = ""
            while True:
                content = fin.readline()
                if content == "":
                    break
                else:
                    for word in content.split():
                        if word != "-1":
                            doc += (word+' ')
            querys.append(doc)
print('FINISH READING QUERYS')



from gensim import corpora
from gensim import corpora, models, similarities

texts = [[word for word in document.split()] for document in documents]


dictionary = corpora.Dictionary(texts)
print(dictionary)
corpus = [dictionary.doc2bow(text) for text in texts]


tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf) # initialize an LSI transformation

q=[]
for qry in querys:
    vec_bow = dictionary.doc2bow(qry.split())
    vec_tfidf = tfidf[vec_bow]
    vec_lsi = lsi[vec_tfidf]
    q.append(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus])

# Compute similarity
print('WAITING TO COMPUTE COSINE SIMILARITY ')
output = ""
for i in range(len(querys)):
    output += ("Query "+str(i+1)+' '+querynames[i]+' '+str(len(documents))+'\n')
    sims = index[q[i]]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for sim in sims:
        output += (filenames[sim[0]]+' '+str(sim[1].item())+'\n')
    output+='\n'
    print('.',end="")
print('\nFINISH COMPUTING COSINE SIMILARITY BETWEEN EACH OTHER')

# Output
with open('ResultsTrainSet.txt','w') as f:
    f.write(output)
print('FINISH OUTPUTING RESULTS')
