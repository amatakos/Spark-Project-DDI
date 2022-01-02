# Name: Alexandros Matakos
# Student ID: 015538460

from pyspark import SparkContext, SparkConf
from pyspark.statcounter import StatCounter
from operator import add
import os
import sys
import random

dataset = "/wrk/group/grp-ddi-2021/datasets/data-1.txt"

conf = (SparkConf()
        .setAppName("amatakos")
        .setMaster("spark://ukko2-10.local.cs.helsinki.fi:7077")
        .set("spark.cores.max", "10")
        .set("spark.rdd.compress", "true")
        .set("spark.broadcast.compress", "true"))
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

data = sc.textFile(dataset)
data = data.map(lambda s: float(s))
data = data.cache()


# Task 1

def statistics(data, with_min_max=False):
        '''
        Calculates the count, mean, variance, min and max for given data.
        Used for Task 1. Also used in each iteration of the binmedian algorithm
        to calculate the mean and variance, to discard elements of the data
        which are more than one standard deviation away from the mean.
        data: rdd, the data.
        with_min_max: boolean, regulates if we also calculate the min and max (we don't
        need to calculate min and max in the algorithm recursions).
        '''
        stats = data.aggregate(StatCounter(), StatCounter.merge, StatCounter.mergeStats)
        count = stats.count()
        mean = stats.mean()
        var = stats.variance()

        if with_min_max:
                min = stats.minValue
                max = stats.maxValue
                return count, min, max, mean, var

        return count, mean, var

full_data_count, min, max, mean, var = statistics(data, with_min_max=True)

print (f"Min value = {min}")
print (f"Max value = {max}")
print (f"Mean = {mean}")
print (f"Variance = {var}")


# Task 2

def binmedian(data, full_data_count, min, max, mean, var):
        '''
        Implements the binmedian algorithm. Uses the fact that the median is
        at most one standard deviation away from the mean, proved by Mallows et. al.,
        to discard values that are more than one sd away from the mean.
        The data is divided into an equally spaced number of bins, which is predetermined
        by the user. The number of elements in each bin is counted, from smallest to largest,
        until more than half of the original data's elements have been amassed. Then the algorithm
        recurses on the last bin found (which is bound to contain the median).
        If the array has even size, the algorithm recurses until it finds the "left" median,
        and then rolls back to the previous step to find the "right" median. This way we avoid
        starting from scratch to compute both medians. This is one of the reasons this implementation
        works much better than the quickselect implementation. One other reason is that in the quickselect
        implementation we cannot leverage Mallows' et. al. result.
        Using full_data_count, min, max, mean, and var as arguments in order to avoid recomputing them,
        since we have already done so in Task 1.
        '''
        N = 20 # number of bins

        current_data = data.filter(lambda x: mean - var**0.5 - 0.01 <= x <= mean + var**0.5 + 0.01).cache()    # discard elements more than 1 sd away from the mean
        current_count = full_data_count           # current iteration's data size
        total_current_left_count = 0    # total elements to the left of current bin
        while current_count > 1:

                I = max - min           # current iteration's data range
                bin_width = I / N       # current iteration's bins width
                spacing = [min + x*bin_width for x in range(N+1)]        # endpoints by which to divide into bins
                bins = data.histogram(spacing)          # create the bins

                bin_sum = total_current_left_count      # start counting from what have on the left so far
                for i, bin_count in enumerate(bins[1]):
                        bin_sum += bin_count            # keep counting until we have more than half of the original data's elements

                        if bin_sum >= full_data_count // 2:   # passed the threshold
                                total_current_left_count_copy = total_current_left_count   # needed for the rollback step to find the other median
                                total_current_left_count = bin_sum - bin_count    # update total_left_count by discarding the element counts that caused us to exceed the threshold, to use it as a basis for the$

                                min = bins[0][i]        # define min and max of the new data on which we will recurse
                                max = bins[0][i+1]

                                if full_data_count % 2 == 0: # if we have to find the right median later, we need a copy of the previous step
                                        copy = current_data.cache()
                                current_data = current_data.filter(lambda x: min <= x <= max).cache()       # select the new data
                                print("\nnew min =", min)
                                print("new max =", max)
                                current_count, current_mean, current_var = statistics(current_data)    # calculate count, mean, variance for the new data
                                print("new data size =", current_count)

                                if current_count > 10000:       # discard elements more than 1 sd away from the current mean (only doing it for large enough data)
                                        current_data = current_data.filter(lambda x: current_mean - current_var**0.5 - 0.01 <= x <= current_mean + current_var**0.5 + 0.01).cache()
                                        current_count, min, max = current_data.count(), current_data.min(), current_data.max()

                                if current_count == 1: # found (left) median
                                        left_median = current_data.first()
                                        if full_data_count % 2 == 1: # if original data is odd sized, we're done.
                                                median = left_median
                                                return median
                                        current_data = copy.cache() # if original data is even sized, roll back one step to find the right median as well
                                        bin_sum = total_current_left_count_copy
                                        for i, bin_count in enumerate(bins[1]):
                                                bin_sum += bin_count
                                                if bin_sum > full_data_count // 2:   # ">=" changed to ">" to find the other median
                                                        min = bins[0][i]
                                                        max = bins[0][i+1]
                                                        right_median = current_data.filter(lambda x: min <= x <= max).first()
                                                        median = 0.5*(left_median + right_median)
                                                        return median
                                break           # break the for loop to recurse on the new data


median = binmedian(data, full_data_count, min, max, mean, var)
print("\nMedian =", median)


# Task 3

# Example showcasing how the mode would be computed
print("\nRunning example to calculate mode")

L = [1,5,1,2,3,5,1,5,10,23,2,5,8]
print("L =",L)

rdd = sc.parallelize(L)

res = (rdd.map(lambda x: (x,1))
          .reduceByKey(lambda x, y: x+y)
          .sortBy(lambda x: x[1], ascending=False)
          .first())
print(f"Mode = {res[0]} with {res[1]} occurences")
