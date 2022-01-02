from pyspark import SparkContext, SparkConf
from operator import add
import numpy as np
import os
import sys
import time

#dataset = "data-2-sample.txt"
dataset = "data-2.txt"

conf = (SparkConf()
        .setAppName("amatakos")
        .setMaster("spark://alex:7077")
        .set("spark.rdd.compress", "true")
        .set("spark.driver.memory", "1g")
        .set("spark.executor.memory", "1g")
        .set("spark.cores.max", "10")
        .set("spark.broadcast.compress", "true"))
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")


def write_answer(answer):
    '''Write answers to answers.txt'''
    with open("answer.txt", "w") as f:
        for s in answer:
            f.write(str(s)+" ")


# read and partition data
data = sc.textFile(dataset, 40)
A = data.map(lambda line: [float(n) for n in line.split()]).cache()


# Multiply A.T * A
AT_A = np.zeros((1000,1000)) # Initialize AT_A. This will hold the result of A.T * A
start = time.time()
# Explanation of the following for loop in the report
for i,partition in enumerate( A.mapPartitions(lambda part: [list(part)]).toLocalIterator() ):
    print(f"\nPartition no. {i+1}/40000")
    for row in partition:
        AT_A += np.outer(row,row)
step1 = time.time()
print(f"\n Time for A^T* * A = {step1-start:.4f}")


# Multiply A*(AT_A)
A_AT_A = A.map(lambda row: np.dot(row, AT_A))

# Exract the first row of A*A.T*A
answer = A_AT_A.first()
print(f"\nFirst row of A_AT_A = {answer[:10]}")

# Write the answers to answers.txt
# write_answer(answer)
