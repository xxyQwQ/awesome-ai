# ======== runMinHashExample =======
# This example code demonstrates comparing documents using the MinHash
# approach.
#
# First, each document is represented by the set of shingles it contains. The
# documents can then be compared using the Jaccard similarity of their
# shingle sets. This is computationally expensive, however, for large numbers
# of documents.
#
# For comparison, we will use the MinHash algorithm to calculate short
# signature vectors to represent the documents. These MinHash signatures can
# then be compared quickly by counting the number of components in which the
# signatures agree. We'll compare all possible pairs of documents, and find
# the pairs with high similarity.

import sys
import time
from hashlib import sha1

# This is the number of components in the resulting MinHash signatures.
# Correspondingly, it is also the number of random hash functions that
# we will need in order to calculate the MinHash.
numHashes = 10

# Read data
numDocs = 1000
dataFile = "../data/articles_1000.train"
truthFile = "../data/articles_1000.truth"

# =============================================================================
#                  Parse The Ground Truth Tables
# =============================================================================
# Build a dictionary mapping the document IDs to their plagiaries, and vice-
# versa.
plagiaries = {}

# Open the truth file.
f = open(truthFile, "rU")

# For each line of the files...
for line in f:
    # Strip the newline character, if present.
    if line[-1] == "\n":
        line = line[0:-1]
    docs = line.split(" ")

    # Map the two documents to each other.
    plagiaries[docs[0]] = docs[1]
    plagiaries[docs[1]] = docs[0]

# =============================================================================
#               Convert Documents To Sets of Shingles
# =============================================================================
print("Shingling articles...")

# Create a dictionary of the articles, mapping the article identifier (e.g.,
# "t8470") to the list of shingle IDs that appear in the document.
docsAsShingleSets = {}

# Open the data file.
f = open(dataFile, "r")

docNames = []
t0 = time.time()
totalShingles = 0

for i in range(0, numDocs):
    # Read all of the words (they are all on one line) and split them by white
    # space.
    words = f.readline().split(" ")

    # Retrieve the article ID, which is the first word on the line.
    docID = words[0]

    # Maintain a list of all document IDs.
    docNames.append(docID)
    del words[0]

    # 'shinglesInDoc' will hold all of the unique shingle IDs present in the
    # current document. If a shingle ID occurs multiple times in the document,
    # it will only appear once in the set (this is a property of Python sets).
    shinglesInDoc = set()

    ##### TODO #####
    # Convert the article into a shingle set
    # Each shingle should contain 3 tokens
    # You should use sha1 (imported from hashlib at the beginning of this file) as the hash function that maps shingles into 32-bit integer.
    # You may use int.from_bytes and hashlib._HASH.digest methods
    for index in range(len(words) - 2):
        shingle = "{} {} {}".format(words[index], words[index + 1], words[index + 2])
        shinglesInDoc.add(int.from_bytes(sha1(shingle.encode()).digest(), byteorder=sys.byteorder))
    ##### end of TODO #####

    # Store the completed list of shingles for this document in the dictionary.
    docsAsShingleSets[docID] = shinglesInDoc

    # Count the number of shingles across all documents.
    totalShingles = totalShingles + (len(words) - 2)

# Close the data file.
f.close()

# Report how long shingling took.
print("\nShingling " + str(numDocs) + " docs took %.2f sec." % (time.time() - t0))
print("\nAverage shingles per doc: %.2f" % (totalShingles / numDocs))

# =============================================================================
#            Define Matrices to Store estimated JSim
# =============================================================================
# define a table `estJSim` to store estimated Jsim
# Hint for efficiency:
# http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
# The Triangular-Matrix Method section
estJSim = [None for _ in range(numDocs * (numDocs - 1) // 2)]

def getTriangleIndex(i, j) -> int:
    if i < j:
        return int(i * (numDocs - (i + 1) / 2.0) + j - i) - 1
    elif i > j:
        return getTriangleIndex(j, i)
    else:
        raise ValueError("Indices must be different")

# =============================================================================
#                 Generate MinHash Signatures
# =============================================================================
# Time this step.
t0 = time.time()
print("\nGenerating random hash functions...")

# Record the maximum shingle ID that we assigned.
maxShingleID = (2 ** 32) - 1

# We need the next largest prime number above 'maxShingleID'.
# http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
nextPrime = 4294967311

# Our random hash function will take the form of: h(x) = (a * x + b) % c
# where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than maxShingleID, i.e. nextPrime
# For convenience, I have randomly picked them and fix them for you.
a_list = [
    2647644122, 3144724950, 1813742201, 3889397089, 850218610,
    4228854938, 3422847010, 1974054350, 1398857723, 3861451095,
]
b_list = [
    2834859619, 3834190079, 3272971987, 1421011856, 1598897977,
    1288507477, 1224561085, 3278591730, 1664131571, 3749293552,
]

# So the i-th hash function is:
# h(x) = (a_list[i] * x + b_list[i]) % nextPrime
print("\nGenerating MinHash signatures for all documents...")

# List of documents represented as signature vectors
signatures = []

# Rather than generating a random permutation of all possible shingles,
# we'll just hash the IDs of the shingles that are actually in the document,
# then take the lowest resulting hash code value. This corresponds to the index
# of the first shingle that you would have encountered in the random order.
for docID in docNames:
    # Get the shingle set for this document.
    shingleIDSet = docsAsShingleSets[docID]

    # The resulting minhash signature for this document. It is supposed to be a list of ints.
    signature = []

    ##### TODO #####
    # complete the signature of this doc
    # i.e. set the list `signature' to a proper value.
    for index in range(numHashes):
        minimum = nextPrime + 1
        for shingleID in shingleIDSet:
            minimum = min(minimum, (a_list[index] * shingleID + b_list[index]) % nextPrime)
        signature.append(minimum)
    ##### end of TODO #####

    # Store the MinHash signature for this document.
    signatures.append(signature)

# Calculate the elapsed time (in seconds)
elapsed = time.time() - t0
print("\nGenerating MinHash signatures took %.2fsec" % elapsed)

# =============================================================================
#                     Compare All Signatures
# =============================================================================
print("\nComparing all signatures...")

# Creates a N x N matrix initialized to 0.
# Time this step.
t0 = time.time()

for i in range(0, numDocs):
    # Get the MinHash signature for document i.
    signature1 = signatures[i]

    for j in range(i + 1, numDocs):
        # Get the MinHash signature for document j.
        signature2 = signatures[j]

        ##### TODO #####
        # calculate the estimated Jaccard Similarity
        # Then store the value into estJSim
        count = 0
        for index in range(numHashes):
            if signature1[index] == signature2[index]:
                count += 1
        estJSim[getTriangleIndex(i, j)] = count / numHashes
        ##### end of TODO #####

# Calculate the elapsed time (in seconds)
elapsed = time.time() - t0
print("\nComparing MinHash signatures took %.2fsec" % elapsed)

# =============================================================================
#                   Display Similar Document Pairs
# =============================================================================
# Count the true positives and false positives.
tp = 0
fp = 0
threshold = 0.5
print("\nList of Document Pairs with J(d1,d2) more than", threshold)
print("Values shown are the estimated Jaccard similarity and the actual")
print("Jaccard similarity.\n")
print("                   Est. J   Act. J")

f = open("../data/prediction.csv", "w")
f.write("article1,article2,Est. J,Act. J\n")

for i in range(0, numDocs):
    for j in range(i + 1, numDocs):
        estJ = estJSim[getTriangleIndex(i, j)]

        # If the similarity is above the threshold...
        if estJ > threshold:
            ##### TODO #####
            # Calculate the actual Jaccard similarity between two docs (shingle sets) for validation.
            set1 = docsAsShingleSets[docNames[i]]
            set2 = docsAsShingleSets[docNames[j]]
            J = len(set1.intersection(set2)) / len(set1.union(set2)) # You should set the actual Jaccard similarity here.
            ##### end of TODO #####

            # Print out the match and similarity values with pretty spacing.
            print("  %5s --> %5s   %.2f     %.2f" % (docNames[i], docNames[j], estJ, J))
            f.write("{},{},{},{}\n".format(docNames[i], docNames[j], estJ, J))

            # Check whether this is a true positive or false positive.
            # We don't need to worry about counting the same true positive twice
            # because we implemented the for-loops to only compare each pair once.
            if plagiaries[docNames[i]] == docNames[j]:
                tp = tp + 1
            else:
                fp = fp + 1

f.close()

# Display true positive and false positive counts.
print()
print("True positives:  " + str(tp) + " / " + str(int(len(plagiaries.keys()) / 2)))
print("False positives: " + str(fp))
