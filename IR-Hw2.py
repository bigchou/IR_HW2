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
print 'FINISH READING DOCUMENTS'

# Remove duplicate values from terms
terms = list(set(terms))

# Build term-position dictionary
dict_terms = {}
for i,v in enumerate(terms):
    dict_terms[v] = i

# Count the term frequency for document matrix
A = np.zeros(shape=(len(terms),len(documents)))
for ind,doc in enumerate(documents):
    for term in doc.split():
        pos = dict_terms[term]
        A[pos][ind]+=1
print 'FINISH COMPUTING TF OF DOCUMENT MATRIX'

# Compute IDF
idf = []
for row in A:
    sum = 0
    for element in row:
        if element > 0:
            sum += 1
    idf.append(log(num_docs / sum))
print 'FINISH COMPUTING IDF'

# Compute TFIDF of document matrix
if(os.path.exists("doc_matrix.dat")):
    A = np.load("doc_matrix.dat")
    print 'FINISH READING TFIDF OF DOCUMENT MATRIX'
else:
    for i in range(len(terms)):
        for j in range(len(documents)):
            if A[i][j] == 0.0:
                A[i][j] = 0.0 * idf[i]
            else:
                A[i][j] = float( 1+log(A[i][j]))  * idf[i]
    A.dump("doc_matrix.dat")
    print 'FINISH COMPUTING TFIDF OF DOCUMENT MATRIX'
    
# ============================================

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
print 'FINISH READING QUERYS'

# Compute TF of query matrix
q = np.zeros(shape=(len(terms),len(querys)))
for ind,qry in enumerate(querys):
    for term in qry.split():
        if term in dict_terms:
            pos = dict_terms[term]
            q[pos][ind]+=1
print 'FINISH COMPUTING TF OF QUERY MATRIX'

# Compute TFIDF of query matirx
for i in range(len(terms)):
    for j in range(len(querys)):
        if q[i][j] == 0.0:
            q[i][j] = 0.0 * idf[i]
        else:
            q[i][j] = float( 1+log(q[i][j])) * idf[i]
print 'FINISH COMPUTING TFIDF OF QUERY MATRIX'

# ============================================

# Compute similarity
print 'WAITING TO COMPUTE COSINE SIMILARITY ',
A = A.transpose()
q = q.transpose()
output = ""
for i in range(len(querys)):
    output += ("Query "+str(i+1)+' '+querynames[i]+' '+str(len(documents))+'\n')
    sim = {}
    for j in range(len(documents)):
        #sim[j] = np.dot(A[j],q[i]) / (np.linalg.norm(A[j]) * np.linalg.norm(q[i]))
        sim[j] = 1 - spatial.distance.cosine(A[j], q[i])
    sim_sort = sorted(sim.items(),key = operator.itemgetter(1),reverse = True)
    for sim in sim_sort:
        output += (filenames[sim[0]]+' '+str(sim[1].item())+'\n')
    output+='\n'
    print '.',
print '\nFINISH COMPUTING COSINE SIMILARITY BETWEEN EACH OTHER'

# Output
with open('ResultsTrainSet.txt','w') as f:
    f.write(output)
print 'FINISH OUTPUTING RESULTS'

